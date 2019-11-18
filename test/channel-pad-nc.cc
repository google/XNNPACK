// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/params.h>

#include "channel-pad-operator-tester.h"


TEST(CHANNEL_PAD_NC_X32, unit_batch_copy) {
  for (size_t input_channels = 1; input_channels < 100; input_channels += 15) {
    ChannelPadOperatorTester()
      .batch_size(1)
      .input_channels(input_channels)
      .iterations(3)
      .TestX32();
  }
}

TEST(CHANNEL_PAD_NC_X32, unit_batch_pad_before) {
  for (size_t input_channels = 1; input_channels < 100; input_channels += 15) {
    for (size_t pad_channels = 1; pad_channels < 50; pad_channels += 7) {
      ChannelPadOperatorTester()
        .batch_size(1)
        .input_channels(input_channels)
        .pad_before(pad_channels)
        .iterations(3)
        .TestX32();
    }
  }
}

TEST(CHANNEL_PAD_NC_X32, unit_batch_pad_after) {
  for (size_t input_channels = 1; input_channels < 100; input_channels += 15) {
    for (size_t pad_channels = 1; pad_channels < 50; pad_channels += 7) {
      ChannelPadOperatorTester()
        .batch_size(1)
        .input_channels(input_channels)
        .pad_after(pad_channels)
        .iterations(3)
        .TestX32();
    }
  }
}

TEST(CHANNEL_PAD_NC_X32, unit_batch_pad_both) {
  for (size_t input_channels = 1; input_channels < 100; input_channels += 15) {
    for (size_t pad_channels = 1; pad_channels < 50; pad_channels += 7) {
      ChannelPadOperatorTester()
        .batch_size(1)
        .input_channels(input_channels)
        .pad_before(pad_channels)
        .pad_after(pad_channels + 1)
        .iterations(3)
        .TestX32();
    }
  }
}

TEST(CHANNEL_PAD_NC_X32, small_batch) {
  for (size_t input_channels = 1; input_channels < 100; input_channels += 15) {
    for (size_t pad_channels = 1; pad_channels < 50; pad_channels += 7) {
      ChannelPadOperatorTester()
        .batch_size(xnn_params.x32.pad.mr)
        .input_channels(input_channels)
        .pad_before(pad_channels)
        .pad_after(pad_channels + 1)
        .iterations(3)
        .TestX32();
    }
  }
}

TEST(CHANNEL_PAD_NC_X32, small_batch_with_x_stride) {
  for (size_t input_channels = 1; input_channels < 100; input_channels += 15) {
    for (size_t pad_channels = 1; pad_channels < 50; pad_channels += 7) {
      ChannelPadOperatorTester()
        .batch_size(xnn_params.x32.pad.mr)
        .input_channels(input_channels)
        .pad_before(pad_channels)
        .pad_after(pad_channels + 1)
        .input_stride(123)
        .iterations(3)
        .TestX32();
    }
  }
}

TEST(CHANNEL_PAD_NC_X32, small_batch_with_y_stride) {
  for (size_t input_channels = 1; input_channels < 100; input_channels += 15) {
    for (size_t pad_channels = 1; pad_channels < 50; pad_channels += 7) {
      ChannelPadOperatorTester()
        .batch_size(xnn_params.x32.pad.mr)
        .input_channels(input_channels)
        .pad_before(pad_channels)
        .pad_after(pad_channels + 1)
        .output_stride(509)
        .iterations(3)
        .TestX32();
    }
  }
}

TEST(CHANNEL_PAD_NC_X32, small_batch_with_x_stride_and_y_stride) {
  for (size_t input_channels = 1; input_channels < 100; input_channels += 15) {
    for (size_t pad_channels = 1; pad_channels < 50; pad_channels += 7) {
      ChannelPadOperatorTester()
        .batch_size(xnn_params.x32.pad.mr)
        .input_channels(input_channels)
        .pad_before(pad_channels)
        .pad_after(pad_channels + 1)
        .input_stride(123)
        .output_stride(509)
        .iterations(3)
        .TestX32();
    }
  }
}

TEST(CHANNEL_PAD_NC_X32, large_batch) {
  for (size_t input_channels = 1; input_channels < 100; input_channels += 15) {
    for (size_t pad_channels = 1; pad_channels < 50; pad_channels += 7) {
      ChannelPadOperatorTester()
        .batch_size(3 * xnn_params.x32.pad.mr + 1)
        .input_channels(input_channels)
        .pad_before(pad_channels)
        .pad_after(pad_channels + 1)
        .iterations(1)
        .TestX32();
    }
  }
}

TEST(CHANNEL_PAD_NC_X32, large_batch_with_x_stride) {
  for (size_t input_channels = 1; input_channels < 100; input_channels += 15) {
    for (size_t pad_channels = 1; pad_channels < 50; pad_channels += 7) {
      ChannelPadOperatorTester()
        .batch_size(3 * xnn_params.x32.pad.mr + 1)
        .input_channels(input_channels)
        .pad_before(pad_channels)
        .pad_after(pad_channels + 1)
        .input_stride(123)
        .iterations(1)
        .TestX32();
    }
  }
}

TEST(CHANNEL_PAD_NC_X32, large_batch_with_y_stride) {
  for (size_t input_channels = 1; input_channels < 100; input_channels += 15) {
    for (size_t pad_channels = 1; pad_channels < 50; pad_channels += 7) {
      ChannelPadOperatorTester()
        .batch_size(3 * xnn_params.x32.pad.mr + 1)
        .input_channels(input_channels)
        .pad_before(pad_channels)
        .pad_after(pad_channels + 1)
        .output_stride(509)
        .iterations(1)
        .TestX32();
    }
  }
}
