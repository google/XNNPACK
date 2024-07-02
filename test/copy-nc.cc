// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>

#include <gtest/gtest.h>
#include "copy-operator-tester.h"

TEST(COPY_NC_X8, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    CopyOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(3)
      .TestX8();
  }
}

TEST(COPY_NC_X8, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    CopyOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestX8();
  }
}

TEST(COPY_NC_X8, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    CopyOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .iterations(3)
      .TestX8();
  }
}

TEST(COPY_NC_X8, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    CopyOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .iterations(3)
      .TestX8();
  }
}

TEST(COPY_NC_X8, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    CopyOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .iterations(3)
      .TestX8();
  }
}


TEST(COPY_NC_X16, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    CopyOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(3)
      .TestX16();
  }
}

TEST(COPY_NC_X16, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    CopyOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestX16();
  }
}

TEST(COPY_NC_X16, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    CopyOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .iterations(3)
      .TestX16();
  }
}

TEST(COPY_NC_X16, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    CopyOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .iterations(3)
      .TestX16();
  }
}

TEST(COPY_NC_X16, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    CopyOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .iterations(3)
      .TestX16();
  }
}


TEST(COPY_NC_X32, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    CopyOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(3)
      .TestX32();
  }
}

TEST(COPY_NC_X32, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    CopyOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestX32();
  }
}

TEST(COPY_NC_X32, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    CopyOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .iterations(3)
      .TestX32();
  }
}

TEST(COPY_NC_X32, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    CopyOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .iterations(3)
      .TestX32();
  }
}

TEST(COPY_NC_X32, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    CopyOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .iterations(3)
      .TestX32();
  }
}
