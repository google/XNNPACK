// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "channel-shuffle-operator-tester.h"


TEST(CHANNEL_SHUFFLE_NC_X8, two_groups_unit_batch) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(1)
      .groups(2)
      .group_channels(group_channels)
      .iterations(3)
      .TestX8();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X8, three_groups_unit_batch) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(1)
      .groups(3)
      .group_channels(group_channels)
      .iterations(3)
      .TestX8();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X8, four_groups_unit_batch) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(1)
      .groups(4)
      .group_channels(group_channels)
      .iterations(3)
      .TestX8();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X8, many_groups_unit_batch) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
      ChannelShuffleOperatorTester()
        .batch_size(1)
        .groups(groups)
        .group_channels(group_channels)
        .iterations(3)
        .TestX8();
    }
  }
}

TEST(CHANNEL_SHUFFLE_NC_X8, two_groups_small_batch) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(2)
      .group_channels(group_channels)
      .iterations(3)
      .TestX8();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X8, three_groups_small_batch) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(3)
      .group_channels(group_channels)
      .iterations(3)
      .TestX8();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X8, four_groups_small_batch) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(4)
      .group_channels(group_channels)
      .iterations(3)
      .TestX8();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X8, many_groups_small_batch) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
      ChannelShuffleOperatorTester()
        .batch_size(3)
        .groups(groups)
        .group_channels(group_channels)
        .iterations(3)
        .TestX8();
    }
  }
}

TEST(CHANNEL_SHUFFLE_NC_X8, two_groups_small_batch_with_input_stride) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(2)
      .group_channels(group_channels)
      .input_stride(511)
      .iterations(3)
      .TestX8();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X8, three_groups_small_batch_with_input_stride) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(3)
      .group_channels(group_channels)
      .input_stride(511)
      .iterations(3)
      .TestX8();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X8, four_groups_small_batch_with_input_stride) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(4)
      .group_channels(group_channels)
      .input_stride(511)
      .iterations(3)
      .TestX8();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X8, many_groups_small_batch_with_input_stride) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
      ChannelShuffleOperatorTester()
        .batch_size(3)
        .groups(groups)
        .group_channels(group_channels)
        .input_stride(1007)
        .iterations(3)
        .TestX8();
    }
  }
}

TEST(CHANNEL_SHUFFLE_NC_X8, two_groups_small_batch_with_output_stride) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(2)
      .group_channels(group_channels)
      .output_stride(513)
      .iterations(3)
      .TestX8();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X8, three_groups_small_batch_with_output_stride) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(3)
      .group_channels(group_channels)
      .output_stride(513)
      .iterations(3)
      .TestX8();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X8, four_groups_small_batch_with_output_stride) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(4)
      .group_channels(group_channels)
      .output_stride(513)
      .iterations(3)
      .TestX8();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X8, many_groups_small_batch_with_output_stride) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
      ChannelShuffleOperatorTester()
        .batch_size(3)
        .groups(groups)
        .group_channels(group_channels)
        .output_stride(1111)
        .iterations(3)
        .TestX8();
    }
  }
}

TEST(CHANNEL_SHUFFLE_NC_X8, two_groups_small_batch_with_input_and_output_stride) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(2)
      .group_channels(group_channels)
      .input_stride(511)
      .output_stride(513)
      .iterations(3)
      .TestX8();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X8, three_groups_small_batch_with_input_and_output_stride) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(3)
      .group_channels(group_channels)
      .input_stride(511)
      .output_stride(513)
      .iterations(3)
      .TestX8();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X8, four_groups_small_batch_with_input_and_output_stride) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(4)
      .group_channels(group_channels)
      .input_stride(511)
      .output_stride(513)
      .iterations(3)
      .TestX8();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X8, many_groups_small_batch_with_input_and_output_stride) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
      ChannelShuffleOperatorTester()
        .batch_size(3)
        .groups(groups)
        .group_channels(group_channels)
        .input_stride(1007)
        .output_stride(1111)
        .iterations(3)
        .TestX8();
    }
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, two_groups_unit_batch) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(1)
      .groups(2)
      .group_channels(group_channels)
      .iterations(3)
      .TestX32();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, three_groups_unit_batch) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(1)
      .groups(3)
      .group_channels(group_channels)
      .iterations(3)
      .TestX32();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, four_groups_unit_batch) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(1)
      .groups(4)
      .group_channels(group_channels)
      .iterations(3)
      .TestX32();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, many_groups_unit_batch) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
      ChannelShuffleOperatorTester()
        .batch_size(1)
        .groups(groups)
        .group_channels(group_channels)
        .iterations(3)
        .TestX32();
    }
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, two_groups_small_batch) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(2)
      .group_channels(group_channels)
      .iterations(3)
      .TestX32();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, three_groups_small_batch) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(3)
      .group_channels(group_channels)
      .iterations(3)
      .TestX32();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, four_groups_small_batch) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(4)
      .group_channels(group_channels)
      .iterations(3)
      .TestX32();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, many_groups_small_batch) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
      ChannelShuffleOperatorTester()
        .batch_size(3)
        .groups(groups)
        .group_channels(group_channels)
        .iterations(3)
        .TestX32();
    }
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, two_groups_small_batch_with_input_stride) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(2)
      .group_channels(group_channels)
      .input_stride(511)
      .iterations(3)
      .TestX32();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, three_groups_small_batch_with_input_stride) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(3)
      .group_channels(group_channels)
      .input_stride(511)
      .iterations(3)
      .TestX32();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, four_groups_small_batch_with_input_stride) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(4)
      .group_channels(group_channels)
      .input_stride(511)
      .iterations(3)
      .TestX32();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, many_groups_small_batch_with_input_stride) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
      ChannelShuffleOperatorTester()
        .batch_size(3)
        .groups(groups)
        .group_channels(group_channels)
        .input_stride(1007)
        .iterations(3)
        .TestX32();
    }
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, two_groups_small_batch_with_output_stride) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(2)
      .group_channels(group_channels)
      .output_stride(513)
      .iterations(3)
      .TestX32();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, three_groups_small_batch_with_output_stride) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(3)
      .group_channels(group_channels)
      .output_stride(513)
      .iterations(3)
      .TestX32();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, four_groups_small_batch_with_output_stride) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(4)
      .group_channels(group_channels)
      .output_stride(513)
      .iterations(3)
      .TestX32();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, many_groups_small_batch_with_output_stride) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
      ChannelShuffleOperatorTester()
        .batch_size(3)
        .groups(groups)
        .group_channels(group_channels)
        .output_stride(1111)
        .iterations(3)
        .TestX32();
    }
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, two_groups_small_batch_with_input_and_output_stride) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(2)
      .group_channels(group_channels)
      .input_stride(511)
      .output_stride(513)
      .iterations(3)
      .TestX32();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, three_groups_small_batch_with_input_and_output_stride) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(3)
      .group_channels(group_channels)
      .input_stride(511)
      .output_stride(513)
      .iterations(3)
      .TestX32();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, four_groups_small_batch_with_input_and_output_stride) {
  for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
    ChannelShuffleOperatorTester()
      .batch_size(3)
      .groups(4)
      .group_channels(group_channels)
      .input_stride(511)
      .output_stride(513)
      .iterations(3)
      .TestX32();
  }
}

TEST(CHANNEL_SHUFFLE_NC_X32, many_groups_small_batch_with_input_and_output_stride) {
  for (size_t groups = 5; groups < 12; groups += 3) {
    for (size_t group_channels = 1; group_channels < 100; group_channels += 15) {
      ChannelShuffleOperatorTester()
        .batch_size(3)
        .groups(groups)
        .group_channels(group_channels)
        .input_stride(1007)
        .output_stride(1111)
        .iterations(3)
        .TestX32();
    }
  }
}
