// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
// Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "test/operators/softmax-operator-tester.h"

#ifndef XNN_EXCLUDE_F16_TESTS
TEST(SOFTMAX_NC_F16, single_class) {
  SoftMaxOperatorTester().batch_size(1).channels(1).iterations(100).TestF16();
}

TEST(SOFTMAX_NC_F16, two_classes) {
  SoftMaxOperatorTester().batch_size(1).channels(2).iterations(100).TestF16();
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
  SoftMaxOperatorTester().batch_size(1).channels(10).iterations(15).TestF16();
  // CIFAR-100
  SoftMaxOperatorTester().batch_size(1).channels(100).iterations(15).TestF16();
}

TEST(SOFTMAX_NC_F16, imagenet_classes) {
  // ImageNet-1K
  SoftMaxOperatorTester().batch_size(1).channels(1000).iterations(10).TestF16();
  // ImageNet-1K+1
  SoftMaxOperatorTester().batch_size(1).channels(1001).iterations(10).TestF16();
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
  SoftMaxOperatorTester().batch_size(1).channels(1).iterations(100).TestF32();
}

TEST(SOFTMAX_NC_F32, two_classes) {
  SoftMaxOperatorTester().batch_size(1).channels(2).iterations(100).TestF32();
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
  SoftMaxOperatorTester().batch_size(1).channels(10).iterations(15).TestF32();
  // CIFAR-100
  SoftMaxOperatorTester().batch_size(1).channels(100).iterations(15).TestF32();
}

TEST(SOFTMAX_NC_F32, imagenet_classes) {
  // ImageNet-1K
  SoftMaxOperatorTester().batch_size(1).channels(1000).iterations(10).TestF32();
  // ImageNet-1K+1
  SoftMaxOperatorTester().batch_size(1).channels(1001).iterations(10).TestF32();
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
  SoftMaxOperatorTester().batch_size(1).channels(1).iterations(100).TestQU8();
}

TEST(SOFTMAX_NC_QU8, two_classes) {
  SoftMaxOperatorTester().batch_size(1).channels(2).iterations(100).TestQU8();
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
  SoftMaxOperatorTester().batch_size(1).channels(10).iterations(15).TestQU8();
  // CIFAR-100
  SoftMaxOperatorTester().batch_size(1).channels(100).iterations(15).TestQU8();
}

TEST(SOFTMAX_NC_QU8, imagenet_classes) {
  // ImageNet-1K
  SoftMaxOperatorTester().batch_size(1).channels(1000).iterations(10).TestQU8();
  // ImageNet-1K+1
  SoftMaxOperatorTester().batch_size(1).channels(1001).iterations(10).TestQU8();
  // ImageNet-22K
  SoftMaxOperatorTester()
      .batch_size(1)
      .channels(21841)
      .iterations(10)
      .TestQU8();
}

TEST(SOFTMAX_NC_QU8, many_channels_with_input_scale) {
  for (size_t channels = 1; channels < 100; channels += 5) {
    for (float input_scale = 1.0e-2f; input_scale < 1.0e+2f;
         input_scale *= 3.14159265f) {
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
    for (int32_t input_zero_point = 0; input_zero_point <= 255;
         input_zero_point += 51) {
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

TEST(SOFTMAX_NC_F32, overflow_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_softmax_nc_f32(/*flags=*/0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  ASSERT_EQ(xnn_status_out_of_memory,
            xnn_reshape_softmax_nc_f32(
                op, /*channels=*/10, /*input_stride=*/SIZE_MAX / 2,
                /*output_stride=*/10, /*batch_size=*/1,
                /*threadpool=*/nullptr));
}

TEST(SOFTMAX_NC_F32, overflow_batch_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_softmax_nc_f32(/*flags=*/0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  ASSERT_EQ(xnn_status_out_of_memory,
            xnn_reshape_softmax_nc_f32(
                op, /*channels=*/10, /*input_stride=*/100,
                /*output_stride=*/100, /*batch_size=*/SIZE_MAX / 50,
                /*threadpool=*/nullptr));
}

TEST(SOFTMAX_NC_QU8, overflow_batch_stride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(/*allocator=*/nullptr));
  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_softmax_nc_qu8(
                /*input_scale=*/1.0f, /*output_zero_point=*/0,
                /*output_scale=*/1.0f / 256.0f, /*flags=*/0, &op));
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_op(
      op, xnn_delete_operator);

  ASSERT_EQ(xnn_status_out_of_memory,
            xnn_reshape_softmax_nc_qu8(
                op, /*channels=*/10, /*input_stride=*/100,
                /*output_stride=*/100, /*batch_size=*/SIZE_MAX / 50,
                /*threadpool=*/nullptr));
}

TEST(SOFTMAX_NC_F32, safe_special_rows) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));
  for (size_t threads : {size_t(1), size_t(4)}) {
    pthreadpool_t pool = pthreadpool_create(threads);
    for (size_t channels :
         {size_t(1), size_t(3), size_t(7), size_t(17), size_t(1500)}) {
      const size_t input_stride = channels + 5;
      const size_t output_stride = channels + 11;
      std::vector<float> input(
          8 * input_stride + XNN_EXTRA_BYTES / sizeof(float), -INFINITY);
      std::vector<float> output(
          8 * output_stride + XNN_EXTRA_BYTES / sizeof(float), 42.0f);
      for (size_t c = 0; c < channels; c++) {
        input[input_stride + c] = float(c) / float(channels);
        input[2 * input_stride + c] = c % 2 ? -INFINITY : 0.0f;
        input[5 * input_stride + c] = c % 2 ? -0.0f : 0.0f;
      }
      input[3 * input_stride + channels / 2] = NAN;
      input[4 * input_stride + channels / 2] = INFINITY;
      input[6 * input_stride] = NAN;
      input[7 * input_stride + channels - 1] = NAN;
      xnn_operator_t op = nullptr;
      ASSERT_EQ(xnn_status_success,
                xnn_create_softmax_nc_f32(XNN_FLAG_SAFE_SOFTMAX, &op));
      for (size_t batch : {size_t(1), size_t(8), size_t(3)}) {
        ASSERT_EQ(xnn_status_success,
                  xnn_reshape_softmax_nc_f32(op, channels, input_stride,
                                             output_stride, batch, pool));
        ASSERT_EQ(xnn_status_success,
                  xnn_setup_softmax_nc_f32(op, input.data(), output.data()));
        ASSERT_EQ(xnn_status_success, xnn_run_operator(op, pool));
        for (size_t b = 0; b < batch; b++) {
          double sum = 0;
          if (b == 1 || b == 2 || b == 5) {
            for (size_t c = 0; c < channels; c++)
              sum += std::exp(double(input[b * input_stride + c]));
          }
          for (size_t c = 0; c < channels; c++) {
            const float actual = output[b * output_stride + c];
            if (b == 0) {
              EXPECT_EQ(actual, 0.0f);
              EXPECT_FALSE(std::signbit(actual));
            } else if (b == 3 || b == 4 || b == 6 || b == 7) {
              EXPECT_TRUE(std::isnan(actual)) << b << ":" << c;
            } else {
              EXPECT_NEAR(actual,
                          std::exp(double(input[b * input_stride + c])) / sum,
                          1.0e-6);
            }
          }
          for (size_t c = channels; c < output_stride; c++)
            EXPECT_EQ(output[b * output_stride + c], 42.0f);
        }
      }
      xnn_delete_operator(op);
    }
    pthreadpool_destroy(pool);
  }
}

TEST(SOFTMAX_NC_F32, ordinary_fully_masked_row) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));
  std::vector<float> input(17 + XNN_EXTRA_BYTES / sizeof(float), -INFINITY);
  std::vector<float> output(input.size());
  xnn_operator_t op = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_softmax_nc_f32(0, &op));
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_softmax_nc_f32(op, 17, 17, 17, 1, nullptr));
  ASSERT_EQ(xnn_status_success,
            xnn_setup_softmax_nc_f32(op, input.data(), output.data()));
  ASSERT_EQ(xnn_status_success, xnn_run_operator(op, nullptr));
  for (size_t c = 0; c < 17; c++) EXPECT_TRUE(std::isnan(output[c]));
  xnn_delete_operator(op);
}

TEST(SOFTMAX_NC_F32, safe_rejects_other_datatypes) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));
  xnn_operator_t op = nullptr;
  EXPECT_EQ(xnn_status_unsupported_parameter,
            xnn_create_softmax_nc_f16(XNN_FLAG_SAFE_SOFTMAX, &op));
  EXPECT_EQ(xnn_status_unsupported_parameter,
            xnn_create_softmax_nc_qu8(1.0f, 0, 0x1.0p-8f, XNN_FLAG_SAFE_SOFTMAX,
                                      &op));
  EXPECT_EQ(op, nullptr);
}
