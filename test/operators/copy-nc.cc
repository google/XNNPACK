// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "test/operators/copy-operator-tester.h"

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

TEST(COPY_NC_X32, InPlaceSetupThenOutOfPlaceSetupStillCopies) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));

  const size_t batch_size = 4;
  const size_t channels = 8;
  std::vector<uint32_t> in_place(batch_size * channels);
  std::vector<uint32_t> source(batch_size * channels);
  std::vector<uint32_t> destination(batch_size * channels, 0);
  for (size_t i = 0; i < source.size(); i++) {
    source[i] = static_cast<uint32_t>(i + 1);
    in_place[i] = source[i];
  }

  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_copy_nc_x32(/*flags=*/0, &op));
  ASSERT_NE(op, nullptr);

  ASSERT_EQ(xnn_status_success,
            xnn_reshape_copy_nc_x32(op, batch_size, channels,
                                    /*input_stride=*/channels,
                                    /*output_stride=*/channels,
                                    /*threadpool=*/nullptr));

  ASSERT_EQ(xnn_status_success,
            xnn_setup_copy_nc_x32(op, in_place.data(), in_place.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  ASSERT_EQ(xnn_status_success,
            xnn_setup_copy_nc_x32(op, source.data(), destination.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, /*threadpool=*/nullptr));

  for (size_t i = 0; i < source.size(); i++) {
    ASSERT_EQ(destination[i], source[i]) << "index " << i;
  }

  EXPECT_EQ(xnn_status_success, xnn_delete_operator(op));
}
