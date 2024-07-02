// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>
#include "softmax-operator-tester.h"

#ifndef XNN_EXCLUDE_F16_TESTS
TEST(SOFTMAX_NC_F16, single_class) {
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(1)
    .iterations(100)
    .TestF16();
}

TEST(SOFTMAX_NC_F16, two_classes) {
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(2)
    .iterations(100)
    .TestF16();
}

TEST(SOFTMAX_NC_F16, many_classes) {
  for (size_t channels = 3; channels < 100; channels++) {
    SoftMaxOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(1)
      .TestF16();
  }
}

TEST(SOFTMAX_NC_F16, cifar_classes) {
  // CIFAR-10
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(10)
    .iterations(15)
    .TestF16();
  // CIFAR-100
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(100)
    .iterations(15)
    .TestF16();
}

TEST(SOFTMAX_NC_F16, imagenet_classes) {
  // ImageNet-1K
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(1000)
    .iterations(10)
    .TestF16();
  // ImageNet-1K+1
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(1001)
    .iterations(10)
    .TestF16();
  // ImageNet-22K
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(21841)
    .iterations(10)
    .TestF16();
}

TEST(SOFTMAX_NC_F16, small_batch) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftMaxOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestF16();
  }
}

TEST(SOFTMAX_NC_F16, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftMaxOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .iterations(3)
      .TestF16();
  }
}

TEST(SOFTMAX_NC_F16, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftMaxOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .iterations(3)
      .TestF16();
  }
}

TEST(SOFTMAX_NC_F16, strided_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftMaxOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .iterations(3)
      .TestF16();
  }
}
#endif  // XNN_EXCLUDE_F16_TESTS


TEST(SOFTMAX_NC_F32, single_class) {
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(1)
    .iterations(100)
    .TestF32();
}

TEST(SOFTMAX_NC_F32, two_classes) {
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(2)
    .iterations(100)
    .TestF32();
}

TEST(SOFTMAX_NC_F32, many_classes) {
  for (size_t channels = 3; channels < 100; channels++) {
    SoftMaxOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(1)
      .TestF32();
  }
}

TEST(SOFTMAX_NC_F32, cifar_classes) {
  // CIFAR-10
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(10)
    .iterations(15)
    .TestF32();
  // CIFAR-100
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(100)
    .iterations(15)
    .TestF32();
}

TEST(SOFTMAX_NC_F32, imagenet_classes) {
  // ImageNet-1K
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(1000)
    .iterations(10)
    .TestF32();
  // ImageNet-1K+1
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(1001)
    .iterations(10)
    .TestF32();
  // ImageNet-22K
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(21841)
    .iterations(10)
    .TestF32();
}

TEST(SOFTMAX_NC_F32, small_batch) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftMaxOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestF32();
  }
}

TEST(SOFTMAX_NC_F32, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftMaxOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .iterations(3)
      .TestF32();
  }
}

TEST(SOFTMAX_NC_F32, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftMaxOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .iterations(3)
      .TestF32();
  }
}

TEST(SOFTMAX_NC_F32, strided_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftMaxOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .iterations(3)
      .TestF32();
  }
}


TEST(SOFTMAX_NC_QU8, single_class) {
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(1)
    .iterations(100)
    .TestQU8();
}

TEST(SOFTMAX_NC_QU8, two_classes) {
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(2)
    .iterations(100)
    .TestQU8();
}

TEST(SOFTMAX_NC_QU8, many_classes) {
  for (size_t channels = 3; channels < 100; channels++) {
    SoftMaxOperatorTester()
      .batch_size(1)
      .channels(channels)
      .iterations(1)
      .TestQU8();
  }
}

TEST(SOFTMAX_NC_QU8, cifar_classes) {
  // CIFAR-10
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(10)
    .iterations(15)
    .TestQU8();
  // CIFAR-100
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(100)
    .iterations(15)
    .TestQU8();
}

TEST(SOFTMAX_NC_QU8, imagenet_classes) {
  // ImageNet-1K
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(1000)
    .iterations(10)
    .TestQU8();
  // ImageNet-1K+1
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(1001)
    .iterations(10)
    .TestQU8();
  // ImageNet-22K
  SoftMaxOperatorTester()
    .batch_size(1)
    .channels(21841)
    .iterations(10)
    .TestQU8();
}

TEST(SOFTMAX_NC_QU8, many_channels_with_input_scale) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    for (float input_scale = 1.0e-2f; input_scale < 1.0e+2f; input_scale *= 3.14159265f) {
      SoftMaxOperatorTester()
        .batch_size(1)
        .channels(channels)
        .input_scale(input_scale)
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(SOFTMAX_NC_QU8, many_channels_with_input_zero_point) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    for (int32_t input_zero_point = 0; input_zero_point <= 255; input_zero_point += 51) {
      SoftMaxOperatorTester()
        .batch_size(1)
        .channels(channels)
        .input_zero_point(uint8_t(input_zero_point))
        .iterations(1)
        .TestQU8();
    }
  }
}

TEST(SOFTMAX_NC_QU8, small_batch) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftMaxOperatorTester()
      .batch_size(3)
      .channels(channels)
      .iterations(3)
      .TestQU8();
  }
}

TEST(SOFTMAX_NC_QU8, small_batch_with_input_stride) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftMaxOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .iterations(3)
      .TestQU8();
  }
}

TEST(SOFTMAX_NC_QU8, small_batch_with_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftMaxOperatorTester()
      .batch_size(3)
      .channels(channels)
      .output_stride(117)
      .iterations(3)
      .TestQU8();
  }
}

TEST(SOFTMAX_NC_QU8, strided_batch_with_input_and_output_stride) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    SoftMaxOperatorTester()
      .batch_size(3)
      .channels(channels)
      .input_stride(129)
      .output_stride(117)
      .iterations(3)
      .TestQU8();
  }
}
