// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <limits>

#include <gtest/gtest.h>

#include "clamp-operator-tester.h"


#ifndef XNN_EXCLUDE_F16_TESTS
TEST(CLAMP_NC_F16, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ClampOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(3)
      .TestF16();
  }
}

TEST(CLAMP_NC_F16, unit_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t qmin = std::numeric_limits<int16_t>::min() + 16;
         qmin < std::numeric_limits<int16_t>::max() - 16;
         qmin += 257)
    { 
      ClampOperatorTester()
        .batch_size(1)
        .channels(channels)
        .qmin(qmin)
        .iterations(3)
        .TestF16();
    }
  }
}

TEST(CLAMP_NC_F16, unit_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t qmax = std::numeric_limits<int16_t>::min() + 16;
         qmax < std::numeric_limits<int16_t>::max() - 16;
         qmax += 257)
    { 
      ClampOperatorTester()
        .batch_size(1)
        .channels(channels)
        .qmax(qmax)
        .iterations(3)
        .TestF16();
    }
  }
}

TEST(CLAMP_NC_F16, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestF16();
  }
}

TEST(CLAMP_NC_F16, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .iterations(3)
      .TestF16();
  }
}

TEST(CLAMP_NC_F16, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .iterations(3)
      .TestF16();
  }
}

TEST(CLAMP_NC_F16, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .iterations(3)
      .TestF16();
  }
}
#endif  // XNN_EXCLUDE_F16_TESTS


TEST(CLAMP_NC_F32, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ClampOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(3)
      .TestF32();
  }
}

TEST(CLAMP_NC_F32, unit_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t qmin = std::numeric_limits<int16_t>::min() + 1;
         qmin < std::numeric_limits<int16_t>::max();
         qmin += 257)
    { 
      ClampOperatorTester()
        .batch_size(1)
        .channels(channels)
        .qmin(qmin)
        .iterations(3)
        .TestF32();
    }
  }
}

TEST(CLAMP_NC_F32, unit_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t qmax = std::numeric_limits<int16_t>::min() + 1;
         qmax < std::numeric_limits<int16_t>::max();
         qmax += 257)
    { 
      ClampOperatorTester()
        .batch_size(1)
        .channels(channels)
        .qmax(qmax)
        .iterations(3)
        .TestF32();
    }
  }
}

TEST(CLAMP_NC_F32, unit_batch_with_relu) {
  for (size_t channels = 1; channels < 100; channels++) {
    ClampOperatorTester()
      .batch_size(1)
      .channels(channels)
      .relu_activation(true)
      .iterations(3)
      .TestF32();
  }
}

TEST(CLAMP_NC_F32, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestF32();
  }
}

TEST(CLAMP_NC_F32, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .iterations(3)
      .TestF32();
  }
}

TEST(CLAMP_NC_F32, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .iterations(3)
      .TestF32();
  }
}

TEST(CLAMP_NC_F32, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .iterations(3)
      .TestF32();
  }
}


TEST(CLAMP_NC_S8, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ClampOperatorTester()
      .batch_size(1)
      .channels(channels)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(3)
      .TestS8();
  }
}

TEST(CLAMP_NC_S8, unit_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t qmin = std::numeric_limits<int8_t>::min() + 1;
         qmin < std::numeric_limits<int8_t>::max();
         qmin++)
    {
      ClampOperatorTester()
        .batch_size(1)
        .channels(channels)
        .qmin(qmin)
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestS8();
    }
  }
}

TEST(CLAMP_NC_S8, unit_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t qmax = std::numeric_limits<int8_t>::min() + 1;
         qmax < std::numeric_limits<int8_t>::max();
         qmax++)
    {
      ClampOperatorTester()
        .batch_size(1)
        .channels(channels)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(qmax)
        .iterations(3)
        .TestS8();
    }
  }
}

TEST(CLAMP_NC_S8, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(3)
      .TestS8();
  }
}

TEST(CLAMP_NC_S8, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(3)
      .TestS8();
  }
}

TEST(CLAMP_NC_S8, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(3)
      .TestS8();
  }
}

TEST(CLAMP_NC_S8, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .qmin(std::numeric_limits<int8_t>::min())
      .qmax(std::numeric_limits<int8_t>::max())
      .iterations(3)
      .TestS8();
  }
}


TEST(CLAMP_NC_U8, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ClampOperatorTester()
      .batch_size(1)
      .channels(channels)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .iterations(3)
      .TestU8();
  }
}

TEST(CLAMP_NC_U8, unit_batch_with_qmin) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (int32_t qmin = std::numeric_limits<uint8_t>::min() + 1;
         qmin < std::numeric_limits<uint8_t>::max();
         qmin++)
    {
      ClampOperatorTester()
        .batch_size(1)
        .channels(channels)
        .qmin(qmin)
        .qmax(std::numeric_limits<uint8_t>::max())
        .iterations(3)
        .TestU8();
    }
  }
}

TEST(CLAMP_NC_U8, unit_batch_with_qmax) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    for (uint8_t qmax = 1; qmax < 255; qmax++) {
      ClampOperatorTester()
        .batch_size(1)
        .channels(channels)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(qmax)
        .iterations(3)
        .TestU8();
    }
  }
}

TEST(CLAMP_NC_U8, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .iterations(3)
      .TestU8();
  }
}

TEST(CLAMP_NC_U8, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .iterations(3)
      .TestU8();
  }
}

TEST(CLAMP_NC_U8, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .iterations(3)
      .TestU8();
  }
}

TEST(CLAMP_NC_U8, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ClampOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .qmin(std::numeric_limits<uint8_t>::min())
      .qmax(std::numeric_limits<uint8_t>::max())
      .iterations(3)
      .TestU8();
  }
}
