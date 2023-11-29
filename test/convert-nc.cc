// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <limits>

#include <gtest/gtest.h>

#include "convert-operator-tester.h"


TEST(CONVERT_NC_F16_F32, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(3)
      .TestF16toF32();
  }
}

TEST(CONVERT_NC_F16_F32, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestF16toF32();
  }
}

TEST(CONVERT_NC_F16_F32, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .iterations(3)
      .TestF16toF32();
  }
}

TEST(CONVERT_NC_F16_F32, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .iterations(3)
      .TestF16toF32();
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
      .TestF16toF32();
  }
}

TEST(CONVERT_NC_F32_F16, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(1)
        .channels(channels)
        .iterations(3)
        .TestF32toF16();
  }
}

TEST(CONVERT_NC_F32_F16, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .iterations(3)
        .TestF32toF16();
  }
}

TEST(CONVERT_NC_F32_F16, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .iterations(3)
        .TestF32toF16();
  }
}

TEST(CONVERT_NC_F32_F16, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .output_stride(117)
        .iterations(3)
        .TestF32toF16();
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
        .TestF32toF16();
  }
}

TEST(CONVERT_NC_F16_QD8, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(1)
        .channels(channels)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestF16toQD8();
  }
}

TEST(CONVERT_NC_F16_QD8, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestF16toQD8();
  }
}

TEST(CONVERT_NC_F16_QD8, small_batch_with_input_stride) {
  for (size_t channels = 10; channels < 11; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestF16toQD8();
  }
}

TEST(CONVERT_NC_F16_QD8, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .output_stride(117)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestF16toQD8();
  }
}

TEST(CONVERT_NC_F16_QD8, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .output_stride(117)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestF16toQD8();
  }
}

TEST(CONVERT_NC_F16_QD8, output_min) {
  for (int16_t qmin = std::numeric_limits<int8_t>::min();
       qmin < std::numeric_limits<int8_t>::max();
       qmin += 51)
  {
    for (size_t channels = 1; channels < 100; channels++) {
      ConvertOperatorTester()
          .batch_size(3)
          .channels(channels)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .iterations(3)
          .TestF16toQD8();
    }
  }
}

TEST(CONVERT_NC_F16_QD8, output_max) {
  for (int16_t qmax = std::numeric_limits<int8_t>::min() + 1;
       qmax <= std::numeric_limits<int8_t>::max();
       qmax += 51)
  {
    for (size_t channels = 1; channels < 100; channels++) {
      ConvertOperatorTester()
          .batch_size(3)
          .channels(channels)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .iterations(3)
          .TestF16toQD8();
    }
  }
}
TEST(CONVERT_NC_F32_QD8, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(1)
        .channels(channels)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestF32toQD8();
  }
}

TEST(CONVERT_NC_F32_QD8, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestF32toQD8();
  }
}

TEST(CONVERT_NC_F32_QD8, small_batch_with_input_stride) {
  for (size_t channels = 10; channels < 11; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestF32toQD8();
  }
}

TEST(CONVERT_NC_F32_QD8, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .output_stride(117)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestF32toQD8();
  }
}

TEST(CONVERT_NC_F32_QD8, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .output_stride(117)
        .qmin(std::numeric_limits<int8_t>::min())
        .qmax(std::numeric_limits<int8_t>::max())
        .iterations(3)
        .TestF32toQD8();
  }
}

TEST(CONVERT_NC_F32_QD8, output_min) {
  for (int16_t qmin = std::numeric_limits<int8_t>::min();
       qmin < std::numeric_limits<int8_t>::max();
       qmin += 51)
  {
    for (size_t channels = 1; channels < 100; channels++) {
      ConvertOperatorTester()
          .batch_size(3)
          .channels(channels)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .iterations(3)
          .TestF32toQD8();
    }
  }
}

TEST(CONVERT_NC_F32_QD8, output_max) {
  for (int16_t qmax = std::numeric_limits<int8_t>::min() + 1;
       qmax <= std::numeric_limits<int8_t>::max();
       qmax += 51)
  {
    for (size_t channels = 1; channels < 100; channels++) {
      ConvertOperatorTester()
          .batch_size(3)
          .channels(channels)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .iterations(3)
          .TestF32toQD8();
    }
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
        .TestF32toQS8();
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
        .TestF32toQS8();
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
        .TestF32toQS8();
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
        .TestF32toQS8();
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
        .TestF32toQS8();
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
          .TestF32toQS8();
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
          .TestF32toQS8();
    }
  }
}

TEST(CONVERT_NC_F32_QS8, output_min) {
  for (int16_t qmin = std::numeric_limits<int8_t>::min();
       qmin < std::numeric_limits<int8_t>::max();
       qmin += 51)
  {
    for (size_t channels = 1; channels < 100; channels++) {
      ConvertOperatorTester()
          .batch_size(3)
          .channels(channels)
          .qmin(qmin)
          .qmax(std::numeric_limits<int8_t>::max())
          .iterations(3)
          .TestF32toQS8();
    }
  }
}

TEST(CONVERT_NC_F32_QS8, output_max) {
  for (int16_t qmax = std::numeric_limits<int8_t>::min() + 1;
       qmax <= std::numeric_limits<int8_t>::max();
       qmax += 51)
  {
    for (size_t channels = 1; channels < 100; channels++) {
      ConvertOperatorTester()
          .batch_size(3)
          .channels(channels)
          .qmin(std::numeric_limits<int8_t>::min())
          .qmax(qmax)
          .iterations(3)
          .TestF32toQS8();
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
        .TestF32toQU8();
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
        .TestF32toQU8();
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
        .TestF32toQU8();
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
        .TestF32toQU8();
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
        .TestF32toQU8();
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
          .TestF32toQU8();
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
          .TestF32toQU8();
    }
  }
}

TEST(CONVERT_NC_F32_QU8, output_min) {
  for (int16_t qmin = std::numeric_limits<uint8_t>::min();
       qmin < std::numeric_limits<uint8_t>::max();
       qmin += 51)
  {
    for (size_t channels = 1; channels < 100; channels++) {
      ConvertOperatorTester()
          .batch_size(3)
          .channels(channels)
          .qmin(qmin)
          .qmax(std::numeric_limits<uint8_t>::max())
          .iterations(3)
          .TestF32toQU8();
    }
  }
}

TEST(CONVERT_NC_F32_QU8, output_max) {
  for (int16_t qmax = std::numeric_limits<uint8_t>::min() + 1;
       qmax <= std::numeric_limits<uint8_t>::max();
       qmax += 51)
  {
    for (size_t channels = 1; channels < 100; channels++) {
      ConvertOperatorTester()
          .batch_size(3)
          .channels(channels)
          .qmin(std::numeric_limits<uint8_t>::min())
          .qmax(qmax)
          .iterations(3)
          .TestF32toQU8();
    }
  }
}

TEST(CONVERT_NC_QS8_F16, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(1)
        .channels(channels)
        .iterations(3)
        .TestQS8toF16();
  }
}

TEST(CONVERT_NC_QS8_F16, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .iterations(3)
        .TestQS8toF16();
  }
}

TEST(CONVERT_NC_QS8_F16, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .iterations(3)
        .TestQS8toF16();
  }
}

TEST(CONVERT_NC_QS8_F16, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .output_stride(117)
        .iterations(3)
        .TestQS8toF16();
  }
}

TEST(CONVERT_NC_QS8_F16, small_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .output_stride(117)
        .iterations(3)
        .TestQS8toF16();
  }
}

TEST(CONVERT_NC_QS8_F16, input_scale) {
  for (float input_scale : std::vector<float>{{0.1f, 1.0f, 10.0f}}) {
    for (size_t channels = 1; channels < 100; channels++) {
      ConvertOperatorTester()
          .batch_size(3)
          .channels(channels)
          .input_scale(input_scale)
          .iterations(3)
          .TestQS8toF16();
    }
  }
}

TEST(CONVERT_NC_QS8_F16, input_zero_point) {
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
          .TestQS8toF16();
    }
  }
}

TEST(CONVERT_NC_QS8_F32, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(1)
        .channels(channels)
        .iterations(3)
        .TestQS8toF32();
  }
}

TEST(CONVERT_NC_QS8_F32, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .iterations(3)
        .TestQS8toF32();
  }
}

TEST(CONVERT_NC_QS8_F32, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .iterations(3)
        .TestQS8toF32();
  }
}

TEST(CONVERT_NC_QS8_F32, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .output_stride(117)
        .iterations(3)
        .TestQS8toF32();
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
        .TestQS8toF32();
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
          .TestQS8toF32();
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
          .TestQS8toF32();
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
        .TestQS16toQS8();
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
        .TestQS16toQS8();
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
        .TestQS16toQS8();
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
        .TestQS16toQS8();
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
        .TestQS16toQS8();
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
          .TestQS16toQS8();
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
          .TestQS16toQS8();
    }
  }
}

TEST(CONVERT_NC_QU8_F32, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(1)
        .channels(channels)
        .iterations(3)
        .TestQU8toF32();
  }
}

TEST(CONVERT_NC_QU8_F32, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .iterations(3)
        .TestQU8toF32();
  }
}

TEST(CONVERT_NC_QU8_F32, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .iterations(3)
        .TestQU8toF32();
  }
}

TEST(CONVERT_NC_QU8_F32, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .output_stride(117)
        .iterations(3)
        .TestQU8toF32();
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
        .TestQU8toF32();
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
          .TestQU8toF32();
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
          .TestQU8toF32();
    }
  }
}
