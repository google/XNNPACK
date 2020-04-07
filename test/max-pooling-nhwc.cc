// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "max-pooling-operator-tester.h"

#include <xnnpack/params.h>


TEST(MAX_POOLING_NHWC_U8, unit_batch_small_1xM_pool) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.u8.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_small_1xM_pool_with_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 3; pool_size <= xnn_params.u8.maxpool.mr; pool_size++) {
      for (size_t padding_left = 0; padding_left <= 1; padding_left++) {
        for (size_t padding_right = 0; padding_right <= 1; padding_right++) {
          MaxPoolingOperatorTester()
            .batch_size(1)
            .input_height(2)
            .input_width(pool_size + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(1)
            .pooling_width(pool_size)
            .channels(channels)
            .TestU8();
        }
      }
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_small_1xM_pool_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 3; pool_size <= xnn_params.u8.maxpool.mr; pool_size++) {
      for (size_t input_width = pool_size; input_width <= pool_size * 2; input_width++) {
        MaxPoolingOperatorTester()
          .batch_size(1)
          .input_height(2)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(1)
          .pooling_width(pool_size)
          .channels(channels)
          .TestU8();
      }
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_small_1xM_pool_with_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.u8.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 4)
        .pooling_height(1)
        .pooling_width(pool_size)
        .stride_width(2)
        .channels(channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_small_1xM_pool_with_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.u8.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(2 * pool_size + 1)
        .padding_left(1)
        .padding_right(1)
        .pooling_height(1)
        .pooling_width(pool_size)
        .dilation_width(2)
        .channels(channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_small_Mx1_pool) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.u8.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_small_Mx1_pool_with_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.u8.maxpool.mr; pool_size++) {
      for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
        for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
          MaxPoolingOperatorTester()
            .batch_size(1)
            .input_height(pool_size + 1)
            .input_width(3)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pool_size)
            .pooling_width(1)
            .channels(channels)
            .TestU8();
        }
      }
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_small_Mx1_pool_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.u8.maxpool.mr; pool_size++) {
      for (size_t input_height = pool_size; input_height <= pool_size * 2; input_height++) {
        MaxPoolingOperatorTester()
          .batch_size(1)
          .input_height(input_height)
          .input_width(3)
          .padding_tf_same(true)
          .pooling_height(pool_size)
          .pooling_width(1)
          .channels(channels)
          .TestU8();
      }
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_small_Mx1_pool_with_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.u8.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 3)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .stride_height(2)
        .channels(channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_small_Mx1_pool_with_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.u8.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2 * pool_size)
        .input_width(3)
        .padding_top(1)
        .padding_bottom(1)
        .pooling_height(pool_size)
        .pooling_width(1)
        .dilation_height(2)
        .channels(channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_small_pool_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.u8.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .input_pixel_stride(5 * channels)
        .TestU8();
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .input_pixel_stride(5 * channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_small_pool_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.u8.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .output_pixel_stride(5 * channels)
        .TestU8();
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .output_pixel_stride(5 * channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_small_pool_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.u8.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .qmin(192)
        .TestU8();
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .qmin(192)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_small_pool_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.u8.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .qmax(192)
        .TestU8();
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .qmax(192)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_large_1xM_pool) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.u8.maxpool.mr + 1; pool_size <= xnn_params.u8.maxpool.mr + xnn_params.u8.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_large_1xM_pool_with_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.u8.maxpool.mr + 1; pool_size <= xnn_params.u8.maxpool.mr + xnn_params.u8.maxpool.qr; pool_size++) {
      for (size_t padding_left = 0; padding_left <= 1; padding_left++) {
        for (size_t padding_right = 0; padding_right <= 1; padding_right++) {
          MaxPoolingOperatorTester()
            .batch_size(1)
            .input_height(2)
            .input_width(pool_size + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(1)
            .pooling_width(pool_size)
            .channels(channels)
            .TestU8();
        }
      }
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_large_1xM_pool_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.u8.maxpool.mr + 1; pool_size <= xnn_params.u8.maxpool.mr + xnn_params.u8.maxpool.qr; pool_size++) {
      for (size_t input_width = pool_size; input_width <= pool_size * 2; input_width++) {
        MaxPoolingOperatorTester()
          .batch_size(1)
          .input_height(2)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(1)
          .pooling_width(pool_size)
          .channels(channels)
          .TestU8();
      }
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_large_1xM_pool_with_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.u8.maxpool.mr + 1; pool_size <= xnn_params.u8.maxpool.mr + xnn_params.u8.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 4)
        .pooling_height(1)
        .pooling_width(pool_size)
        .stride_width(2)
        .channels(channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_large_1xM_pool_with_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.u8.maxpool.mr + 1; pool_size <= xnn_params.u8.maxpool.mr + xnn_params.u8.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(2 * pool_size + 1)
        .padding_left(1)
        .padding_right(1)
        .pooling_height(1)
        .pooling_width(pool_size)
        .dilation_width(2)
        .channels(channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_large_Mx1_pool) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.u8.maxpool.mr + 1; pool_size <= xnn_params.u8.maxpool.mr + xnn_params.u8.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_large_Mx1_pool_with_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.u8.maxpool.mr + 1; pool_size <= xnn_params.u8.maxpool.mr + xnn_params.u8.maxpool.qr; pool_size++) {
      for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
        for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
          MaxPoolingOperatorTester()
            .batch_size(1)
            .input_height(pool_size + 1)
            .input_width(3)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pool_size)
            .pooling_width(1)
            .channels(channels)
            .TestU8();
        }
      }
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_large_Mx1_pool_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.u8.maxpool.mr + 1; pool_size <= xnn_params.u8.maxpool.mr + xnn_params.u8.maxpool.qr; pool_size++) {
      for (size_t input_height = pool_size; input_height <= pool_size * 2; input_height++) {
        MaxPoolingOperatorTester()
          .batch_size(1)
          .input_height(input_height)
          .input_width(3)
          .padding_tf_same(true)
          .pooling_height(pool_size)
          .pooling_width(1)
          .channels(channels)
          .TestU8();
      }
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_large_Mx1_pool_with_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.u8.maxpool.mr + 1; pool_size <= xnn_params.u8.maxpool.mr + xnn_params.u8.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 3)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .stride_height(2)
        .channels(channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_large_Mx1_pool_with_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.u8.maxpool.mr + 1; pool_size <= xnn_params.u8.maxpool.mr + xnn_params.u8.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2 * pool_size)
        .input_width(3)
        .padding_top(1)
        .padding_bottom(1)
        .pooling_height(pool_size)
        .pooling_width(1)
        .dilation_height(2)
        .channels(channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_large_pool_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.u8.maxpool.mr + 1; pool_size <= xnn_params.u8.maxpool.mr + xnn_params.u8.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .input_pixel_stride(5 * channels)
        .TestU8();
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .input_pixel_stride(5 * channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_large_pool_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.u8.maxpool.mr + 1; pool_size <= xnn_params.u8.maxpool.mr + xnn_params.u8.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .output_pixel_stride(5 * channels)
        .TestU8();
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .output_pixel_stride(5 * channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_large_pool_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.u8.maxpool.mr + 1; pool_size <= xnn_params.u8.maxpool.mr + xnn_params.u8.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .qmin(192)
        .TestU8();
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .qmin(192)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, unit_batch_large_pool_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.u8.maxpool.mr + 1; pool_size <= xnn_params.u8.maxpool.mr + xnn_params.u8.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .qmax(192)
        .TestU8();
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .qmax(192)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, small_batch_small_pool) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.u8.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .TestU8();
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, small_batch_small_pool_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.u8.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .input_pixel_stride(5 * channels)
        .TestU8();
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .input_pixel_stride(5 * channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, small_batch_small_pool_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.u8.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .output_pixel_stride(5 * channels)
        .TestU8();
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .output_pixel_stride(5 * channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, small_batch_large_pool) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.u8.maxpool.mr + 1; pool_size <= xnn_params.u8.maxpool.mr + xnn_params.u8.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .TestU8();
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, small_batch_large_pool_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.u8.maxpool.mr + 1; pool_size <= xnn_params.u8.maxpool.mr + xnn_params.u8.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .input_pixel_stride(5 * channels)
        .TestU8();
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .input_pixel_stride(5 * channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, small_batch_large_pool_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.u8.maxpool.mr + 1; pool_size <= xnn_params.u8.maxpool.mr + xnn_params.u8.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .output_pixel_stride(5 * channels)
        .TestU8();
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .output_pixel_stride(5 * channels)
        .TestU8();
    }
  }
}

TEST(MAX_POOLING_NHWC_U8, setup_increasing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  MaxPoolingOperatorTester()
    .batch_size(3)
    .next_batch_size(5)
    .input_height(8)
    .input_width(8)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupU8();
}

TEST(MAX_POOLING_NHWC_U8, setup_decreasing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  MaxPoolingOperatorTester()
    .batch_size(5)
    .next_batch_size(3)
    .input_height(8)
    .input_width(8)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupU8();
}

TEST(MAX_POOLING_NHWC_U8, setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  MaxPoolingOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(9)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupU8();
  MaxPoolingOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(7)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupU8();
}

TEST(MAX_POOLING_NHWC_U8, setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  MaxPoolingOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(9)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupU8();
  MaxPoolingOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(7)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupU8();
}

TEST(MAX_POOLING_NHWC_U8, setup_swap_height_and_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  MaxPoolingOperatorTester()
    .batch_size(3)
    .input_height(9)
    .input_width(8)
    .next_input_height(8)
    .next_input_width(9)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupU8();
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_small_1xM_pool) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.f32.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_small_1xM_pool_with_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 3; pool_size <= xnn_params.f32.maxpool.mr; pool_size++) {
      for (size_t padding_left = 0; padding_left <= 1; padding_left++) {
        for (size_t padding_right = 0; padding_right <= 1; padding_right++) {
          MaxPoolingOperatorTester()
            .batch_size(1)
            .input_height(2)
            .input_width(pool_size + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(1)
            .pooling_width(pool_size)
            .channels(channels)
            .TestF32();
        }
      }
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_small_1xM_pool_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 3; pool_size <= xnn_params.f32.maxpool.mr; pool_size++) {
      for (size_t input_width = pool_size; input_width <= pool_size * 2; input_width++) {
        MaxPoolingOperatorTester()
          .batch_size(1)
          .input_height(2)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(1)
          .pooling_width(pool_size)
          .channels(channels)
          .TestF32();
      }
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_small_1xM_pool_with_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.f32.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 4)
        .pooling_height(1)
        .pooling_width(pool_size)
        .stride_width(2)
        .channels(channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_small_1xM_pool_with_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.f32.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(2 * pool_size + 1)
        .padding_left(1)
        .padding_right(1)
        .pooling_height(1)
        .pooling_width(pool_size)
        .dilation_width(2)
        .channels(channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_small_Mx1_pool) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.f32.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_small_Mx1_pool_with_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.f32.maxpool.mr; pool_size++) {
      for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
        for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
          MaxPoolingOperatorTester()
            .batch_size(1)
            .input_height(pool_size + 1)
            .input_width(3)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pool_size)
            .pooling_width(1)
            .channels(channels)
            .TestF32();
        }
      }
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_small_Mx1_pool_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.f32.maxpool.mr; pool_size++) {
      for (size_t input_height = pool_size; input_height <= pool_size * 2; input_height++) {
        MaxPoolingOperatorTester()
          .batch_size(1)
          .input_height(input_height)
          .input_width(3)
          .padding_tf_same(true)
          .pooling_height(pool_size)
          .pooling_width(1)
          .channels(channels)
          .TestF32();
      }
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_small_Mx1_pool_with_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.f32.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 3)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .stride_height(2)
        .channels(channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_small_Mx1_pool_with_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.f32.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2 * pool_size)
        .input_width(3)
        .padding_top(1)
        .padding_bottom(1)
        .pooling_height(pool_size)
        .pooling_width(1)
        .dilation_height(2)
        .channels(channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_small_pool_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.f32.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .input_pixel_stride(5 * channels)
        .TestF32();
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .input_pixel_stride(5 * channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_small_pool_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.f32.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .output_pixel_stride(5 * channels)
        .TestF32();
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .output_pixel_stride(5 * channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_small_pool_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.f32.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .qmin(192)
        .TestF32();
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .qmin(192)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_small_pool_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.f32.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .qmax(192)
        .TestF32();
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .qmax(192)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_large_1xM_pool) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.f32.maxpool.mr + 1; pool_size <= xnn_params.f32.maxpool.mr + xnn_params.f32.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_large_1xM_pool_with_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.f32.maxpool.mr + 1; pool_size <= xnn_params.f32.maxpool.mr + xnn_params.f32.maxpool.qr; pool_size++) {
      for (size_t padding_left = 0; padding_left <= 1; padding_left++) {
        for (size_t padding_right = 0; padding_right <= 1; padding_right++) {
          MaxPoolingOperatorTester()
            .batch_size(1)
            .input_height(2)
            .input_width(pool_size + 2)
            .padding_left(padding_left)
            .padding_right(padding_right)
            .pooling_height(1)
            .pooling_width(pool_size)
            .channels(channels)
            .TestF32();
        }
      }
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_large_1xM_pool_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.f32.maxpool.mr + 1; pool_size <= xnn_params.f32.maxpool.mr + xnn_params.f32.maxpool.qr; pool_size++) {
      for (size_t input_width = pool_size; input_width <= pool_size * 2; input_width++) {
        MaxPoolingOperatorTester()
          .batch_size(1)
          .input_height(2)
          .input_width(input_width)
          .padding_tf_same(true)
          .pooling_height(1)
          .pooling_width(pool_size)
          .channels(channels)
          .TestF32();
      }
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_large_1xM_pool_with_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.f32.maxpool.mr + 1; pool_size <= xnn_params.f32.maxpool.mr + xnn_params.f32.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 4)
        .pooling_height(1)
        .pooling_width(pool_size)
        .stride_width(2)
        .channels(channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_large_1xM_pool_with_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.f32.maxpool.mr + 1; pool_size <= xnn_params.f32.maxpool.mr + xnn_params.f32.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(2 * pool_size + 1)
        .padding_left(1)
        .padding_right(1)
        .pooling_height(1)
        .pooling_width(pool_size)
        .dilation_width(2)
        .channels(channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_large_Mx1_pool) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.f32.maxpool.mr + 1; pool_size <= xnn_params.f32.maxpool.mr + xnn_params.f32.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_large_Mx1_pool_with_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.f32.maxpool.mr + 1; pool_size <= xnn_params.f32.maxpool.mr + xnn_params.f32.maxpool.qr; pool_size++) {
      for (size_t padding_top = 0; padding_top <= 1; padding_top++) {
        for (size_t padding_bottom = 0; padding_bottom <= 1; padding_bottom++) {
          MaxPoolingOperatorTester()
            .batch_size(1)
            .input_height(pool_size + 1)
            .input_width(3)
            .padding_top(padding_top)
            .padding_bottom(padding_bottom)
            .pooling_height(pool_size)
            .pooling_width(1)
            .channels(channels)
            .TestF32();
        }
      }
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_large_Mx1_pool_with_tf_same_padding) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.f32.maxpool.mr + 1; pool_size <= xnn_params.f32.maxpool.mr + xnn_params.f32.maxpool.qr; pool_size++) {
      for (size_t input_height = pool_size; input_height <= pool_size * 2; input_height++) {
        MaxPoolingOperatorTester()
          .batch_size(1)
          .input_height(input_height)
          .input_width(3)
          .padding_tf_same(true)
          .pooling_height(pool_size)
          .pooling_width(1)
          .channels(channels)
          .TestF32();
      }
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_large_Mx1_pool_with_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.f32.maxpool.mr + 1; pool_size <= xnn_params.f32.maxpool.mr + xnn_params.f32.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 3)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .stride_height(2)
        .channels(channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_large_Mx1_pool_with_dilation) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.f32.maxpool.mr + 1; pool_size <= xnn_params.f32.maxpool.mr + xnn_params.f32.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2 * pool_size)
        .input_width(3)
        .padding_top(1)
        .padding_bottom(1)
        .pooling_height(pool_size)
        .pooling_width(1)
        .dilation_height(2)
        .channels(channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_large_pool_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.f32.maxpool.mr + 1; pool_size <= xnn_params.f32.maxpool.mr + xnn_params.f32.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .input_pixel_stride(5 * channels)
        .TestF32();
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .input_pixel_stride(5 * channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_large_pool_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.f32.maxpool.mr + 1; pool_size <= xnn_params.f32.maxpool.mr + xnn_params.f32.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .output_pixel_stride(5 * channels)
        .TestF32();
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .output_pixel_stride(5 * channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_large_pool_with_qmin) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.f32.maxpool.mr + 1; pool_size <= xnn_params.f32.maxpool.mr + xnn_params.f32.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .qmin(192)
        .TestF32();
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .qmin(192)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, unit_batch_large_pool_with_qmax) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.f32.maxpool.mr + 1; pool_size <= xnn_params.f32.maxpool.mr + xnn_params.f32.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .qmax(192)
        .TestF32();
      MaxPoolingOperatorTester()
        .batch_size(1)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .qmax(192)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, small_batch_small_pool) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.f32.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .TestF32();
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, small_batch_small_pool_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.f32.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .input_pixel_stride(5 * channels)
        .TestF32();
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .input_pixel_stride(5 * channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, small_batch_small_pool_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = 2; pool_size <= xnn_params.f32.maxpool.mr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .output_pixel_stride(5 * channels)
        .TestF32();
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .output_pixel_stride(5 * channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, small_batch_large_pool) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.f32.maxpool.mr + 1; pool_size <= xnn_params.f32.maxpool.mr + xnn_params.f32.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .TestF32();
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, small_batch_large_pool_with_input_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.f32.maxpool.mr + 1; pool_size <= xnn_params.f32.maxpool.mr + xnn_params.f32.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .input_pixel_stride(5 * channels)
        .TestF32();
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .input_pixel_stride(5 * channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, small_batch_large_pool_with_output_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  for (size_t channels = 1; channels <= 100; channels += 15) {
    for (size_t pool_size = xnn_params.f32.maxpool.mr + 1; pool_size <= xnn_params.f32.maxpool.mr + xnn_params.f32.maxpool.qr; pool_size++) {
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(pool_size + 1)
        .input_width(3)
        .pooling_height(pool_size)
        .pooling_width(1)
        .channels(channels)
        .output_pixel_stride(5 * channels)
        .TestF32();
      MaxPoolingOperatorTester()
        .batch_size(3)
        .input_height(2)
        .input_width(pool_size + 2)
        .pooling_height(1)
        .pooling_width(pool_size)
        .channels(channels)
        .output_pixel_stride(5 * channels)
        .TestF32();
    }
  }
}

TEST(MAX_POOLING_NHWC_F32, setup_increasing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  MaxPoolingOperatorTester()
    .batch_size(3)
    .next_batch_size(5)
    .input_height(8)
    .input_width(8)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF32();
}

TEST(MAX_POOLING_NHWC_F32, setup_decreasing_batch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  MaxPoolingOperatorTester()
    .batch_size(5)
    .next_batch_size(3)
    .input_height(8)
    .input_width(8)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF32();
}

TEST(MAX_POOLING_NHWC_F32, setup_changing_height) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  MaxPoolingOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(9)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF32();
  MaxPoolingOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_height(7)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF32();
}

TEST(MAX_POOLING_NHWC_F32, setup_changing_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  MaxPoolingOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(9)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF32();
  MaxPoolingOperatorTester()
    .batch_size(3)
    .input_height(8)
    .input_width(8)
    .next_input_width(7)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF32();
}

TEST(MAX_POOLING_NHWC_F32, setup_swap_height_and_width) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  MaxPoolingOperatorTester()
    .batch_size(3)
    .input_height(9)
    .input_width(8)
    .next_input_height(8)
    .next_input_width(9)
    .pooling_height(5)
    .pooling_width(3)
    .channels(24)
    .TestSetupF32();
}
