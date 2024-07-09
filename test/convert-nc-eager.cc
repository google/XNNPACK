// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "convert-operator-tester.h"

TEST(CONVERT_NC_F16_F32, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(3)
      .TestRunF16toF32();
  }
}

TEST(CONVERT_NC_F16_F32, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestRunF16toF32();
  }
}

TEST(CONVERT_NC_F16_F32, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .iterations(3)
      .TestRunF16toF32();
  }
}

TEST(CONVERT_NC_F16_F32, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .iterations(3)
      .TestRunF16toF32();
  }
}

TEST(CONVERT_NC_F16_F32, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .iterations(3)
      .TestRunF16toF32();
  }
}

TEST(CONVERT_NC_F32_F16, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(1)
        .channels(channels)
        .iterations(3)
        .TestRunF32toF16();
  }
}

TEST(CONVERT_NC_F32_F16, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .iterations(3)
        .TestRunF32toF16();
  }
}

TEST(CONVERT_NC_F32_F16, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .iterations(3)
        .TestRunF32toF16();
  }
}

TEST(CONVERT_NC_F32_F16, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .output_stride(117)
        .iterations(3)
        .TestRunF32toF16();
  }
}

TEST(CONVERT_NC_F32_F16, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .output_stride(117)
        .iterations(3)
        .TestRunF32toF16();
  }
}

TEST(CONVERT_NC_F32_QS8, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(1)
        .channels(channels)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestRunF32toQS8();
  }
}

TEST(CONVERT_NC_F32_QS8, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestRunF32toQS8();
  }
}

TEST(CONVERT_NC_F32_QS8, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestRunF32toQS8();
  }
}

TEST(CONVERT_NC_F32_QS8, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .output_stride(117)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestRunF32toQS8();
  }
}

TEST(CONVERT_NC_F32_QS8, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .output_stride(117)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestRunF32toQS8();
  }
}

TEST(CONVERT_NC_F32_QS8, output_scale) {
  for (float output_scale : std::vector<float>{{0.1f, 1.0f, 10.0f}}) {
    for (size_t channels = 1; channels < 100; channels++) {
      ConvertOperatorTester()
          .batch_size(3)
          .channels(channels)
          .output_scale(output_scale)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .iterations(3)
          .TestRunF32toQS8();
    }
  }
}

TEST(CONVERT_NC_F32_QS8, output_zero_point) {
  for (int16_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point += 51)
  {
    for (size_t channels = 1; channels < 100; channels++) {
      ConvertOperatorTester()
          .batch_size(3)
          .channels(channels)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .iterations(3)
          .TestRunF32toQS8();
    }
  }
}

TEST(CONVERT_NC_QS8_F32, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(1)
        .channels(channels)
        .iterations(3)
        .TestRunQS8toF32();
  }
}

TEST(CONVERT_NC_QS8_F32, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .iterations(3)
        .TestRunQS8toF32();
  }
}

TEST(CONVERT_NC_QS8_F32, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .iterations(3)
        .TestRunQS8toF32();
  }
}

TEST(CONVERT_NC_QS8_F32, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .output_stride(117)
        .iterations(3)
        .TestRunQS8toF32();
  }
}

TEST(CONVERT_NC_QS8_F32, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .output_stride(117)
        .iterations(3)
        .TestRunQS8toF32();
  }
}

TEST(CONVERT_NC_QS8_F32, input_scale) {
  for (float input_scale : std::vector<float>{{0.1f, 1.0f, 10.0f}}) {
    for (size_t channels = 1; channels < 100; channels++) {
      ConvertOperatorTester()
          .batch_size(3)
          .channels(channels)
          .input_scale(input_scale)
          .iterations(3)
          .TestRunQS8toF32();
    }
  }
}

TEST(CONVERT_NC_QS8_F32, input_zero_point) {
  for (int16_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point += 51)
  {
    for (size_t channels = 1; channels < 100; channels++) {
      ConvertOperatorTester()
          .batch_size(3)
          .channels(channels)
          .zero_point(zero_point)
          .iterations(3)
          .TestRunQS8toF32();
    }
  }
}

TEST(CONVERT_NC_QS16_QS8, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(1)
        .channels(channels)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestRunQS16toQS8();
  }
}

TEST(CONVERT_NC_QS16_QS8, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestRunQS16toQS8();
  }
}

TEST(CONVERT_NC_QS16_QS8, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestRunQS16toQS8();
  }
}

TEST(CONVERT_NC_QS16_QS8, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .output_stride(117)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestRunQS16toQS8();
  }
}

TEST(CONVERT_NC_QS16_QS8, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .output_stride(117)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestRunQS16toQS8();
  }
}

TEST(CONVERT_NC_QS16_QS8, input_scale) {
  for (float input_scale : std::vector<float>{{0.1f, 1.0f, 10.0f}}) {
    for (size_t channels = 1; channels < 100; channels++) {
      ConvertOperatorTester()
          .batch_size(3)
          .channels(channels)
          .input_scale(input_scale)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .iterations(3)
          .TestRunQS16toQS8();
    }
  }
}

TEST(CONVERT_NC_QS16_QS8, output_zero_point) {
  for (int16_t zero_point = std::numeric_limits<int8_t>::min();
       zero_point <= std::numeric_limits<int8_t>::max();
       zero_point += 51)
  {
    for (size_t channels = 1; channels < 100; channels++) {
      ConvertOperatorTester()
          .batch_size(3)
          .channels(channels)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(std::numeric_limits<int8_t>::max())
          .iterations(3)
          .TestRunQS16toQS8();
    }
  }
}

TEST(CONVERT_NC_F32_QU8, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(1)
        .channels(channels)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .iterations(3)
        .TestRunF32toQU8();
  }
}

TEST(CONVERT_NC_F32_QU8, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .iterations(3)
        .TestRunF32toQU8();
  }
}

TEST(CONVERT_NC_F32_QU8, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .iterations(3)
        .TestRunF32toQU8();
  }
}

TEST(CONVERT_NC_F32_QU8, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .output_stride(117)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .iterations(3)
        .TestRunF32toQU8();
  }
}

TEST(CONVERT_NC_F32_QU8, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .output_stride(117)
        .qmin(std::numeric_limits<uint8_t>::min())
        .qmax(std::numeric_limits<uint8_t>::max())
        .iterations(3)
        .TestRunF32toQU8();
  }
}

TEST(CONVERT_NC_F32_QU8, output_scale) {
  for (float output_scale : std::vector<float>{{0.1f, 1.0f, 10.0f}}) {
    for (size_t channels = 1; channels < 100; channels++) {
      ConvertOperatorTester()
          .batch_size(3)
          .channels(channels)
          .output_scale(output_scale)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .iterations(3)
          .TestRunF32toQU8();
    }
  }
}

TEST(CONVERT_NC_F32_QU8, output_zero_point) {
  for (int16_t zero_point = std::numeric_limits<uint8_t>::min();
       zero_point <= std::numeric_limits<uint8_t>::max();
       zero_point += 51)
  {
    for (size_t channels = 1; channels < 100; channels++) {
      ConvertOperatorTester()
          .batch_size(3)
          .channels(channels)
          .zero_point(zero_point)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(std::numeric_limits<uint8_t>::max())
          .iterations(3)
          .TestRunF32toQU8();
    }
  }
}

TEST(CONVERT_NC_QU8_F32, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(1)
        .channels(channels)
        .iterations(3)
        .TestRunQU8toF32();
  }
}

TEST(CONVERT_NC_QU8_F32, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .iterations(3)
        .TestRunQU8toF32();
  }
}

TEST(CONVERT_NC_QU8_F32, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .iterations(3)
        .TestRunQU8toF32();
  }
}

TEST(CONVERT_NC_QU8_F32, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .output_stride(117)
        .iterations(3)
        .TestRunQU8toF32();
  }
}

TEST(CONVERT_NC_QU8_F32, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .output_stride(117)
        .iterations(3)
        .TestRunQU8toF32();
  }
}

TEST(CONVERT_NC_QU8_F32, input_scale) {
  for (float input_scale : std::vector<float>{{0.1f, 1.0f, 10.0f}}) {
    for (size_t channels = 1; channels < 100; channels++) {
      ConvertOperatorTester()
          .batch_size(3)
          .channels(channels)
          .input_scale(input_scale)
          .iterations(3)
          .TestRunQU8toF32();
    }
  }
}

TEST(CONVERT_NC_QU8_F32, input_zero_point) {
  for (int16_t zero_point = std::numeric_limits<uint8_t>::min();
       zero_point <= std::numeric_limits<uint8_t>::max();
       zero_point += 51)
  {
    for (size_t channels = 1; channels < 100; channels++) {
      ConvertOperatorTester()
          .batch_size(3)
          .channels(channels)
          .zero_point(zero_point)
          .iterations(3)
          .TestRunQU8toF32();
    }
  }
}
