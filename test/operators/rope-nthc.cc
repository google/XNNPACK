// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "test/operators/rope-operator-tester.h"

TEST(ROPE_NTHC_F16, two_channels) {
  RoPEOperatorTester().batch_size(1).heads(1).tokens(1).channels(2).TestF16();
}

TEST(ROPE_NTHC_F16, multiple_channels) {
  RoPEOperatorTester().batch_size(1).heads(1).tokens(1).channels(42).TestF16();
}

TEST(ROPE_NTHC_F16, multiple_tokens) {
  RoPEOperatorTester().batch_size(1).heads(1).tokens(11).channels(42).TestF16();
}

TEST(ROPE_NTHC_F16, multiple_heads) {
  RoPEOperatorTester().batch_size(1).heads(7).tokens(11).channels(42).TestF16();
}

TEST(ROPE_NTHC_F16, nonunit_batch) {
  RoPEOperatorTester().batch_size(3).heads(7).tokens(11).channels(42).TestF16();
}

TEST(ROPE_NTHC_F32, two_channels) {
  RoPEOperatorTester().batch_size(1).heads(1).tokens(1).channels(2).TestF32();
}

TEST(ROPE_NTHC_F32, multiple_channels) {
  RoPEOperatorTester().batch_size(1).heads(1).tokens(1).channels(42).TestF32();
}

TEST(ROPE_NTHC_F32, multiple_tokens) {
  RoPEOperatorTester().batch_size(1).heads(1).tokens(11).channels(42).TestF32();
}

TEST(ROPE_NTHC_F32, multiple_heads) {
  RoPEOperatorTester().batch_size(1).heads(7).tokens(11).channels(42).TestF32();
}

TEST(ROPE_NTHC_F32, nonunit_batch) {
  RoPEOperatorTester().batch_size(3).heads(7).tokens(11).channels(42).TestF32();
}

TEST(ROPE_NTHC_F16, sequence_stride_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  xnn_operator_t rope_op = nullptr;
  const xnn_status status = xnn_create_rope_nthc_f16(/*flags=*/0, &rope_op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, rope_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_rope_op(
      rope_op, xnn_delete_operator);

  const size_t batch_size = 1;
  const size_t tokens = 1;
  const size_t channels = 4;
  const size_t heads = SIZE_MAX / 2 + 1;

  ASSERT_EQ(xnn_status_out_of_memory,
            xnn_reshape_rope_nthc_f16(rope_op, batch_size, tokens, heads,
                                      channels, /*threadpool=*/nullptr));
}

TEST(ROPE_NTHC_F16, batch_stride_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  xnn_operator_t rope_op = nullptr;
  const xnn_status status = xnn_create_rope_nthc_f16(/*flags=*/0, &rope_op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, rope_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_rope_op(
      rope_op, xnn_delete_operator);

  const size_t batch_size = 1;
  const size_t tokens = SIZE_MAX / 2 + 1;
  const size_t heads = 2;
  const size_t channels = 2;

  ASSERT_EQ(xnn_status_out_of_memory,
            xnn_reshape_rope_nthc_f16(rope_op, batch_size, tokens, heads,
                                      channels, /*threadpool=*/nullptr));
}

TEST(ROPE_NTHC_F32, sequence_stride_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  xnn_operator_t rope_op = nullptr;
  const xnn_status status = xnn_create_rope_nthc_f32(/*flags=*/0, &rope_op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, rope_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_rope_op(
      rope_op, xnn_delete_operator);

  const size_t batch_size = 1;
  const size_t tokens = 1;
  const size_t channels = 4;
  const size_t heads = SIZE_MAX / 2 + 1;

  ASSERT_EQ(xnn_status_out_of_memory,
            xnn_reshape_rope_nthc_f32(rope_op, batch_size, tokens, heads,
                                      channels, /*threadpool=*/nullptr));
}

TEST(ROPE_NTHC_F32, batch_stride_overflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  xnn_operator_t rope_op = nullptr;
  const xnn_status status = xnn_create_rope_nthc_f32(/*flags=*/0, &rope_op);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  ASSERT_NE(nullptr, rope_op);
  std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_rope_op(
      rope_op, xnn_delete_operator);

  const size_t batch_size = 1;
  const size_t tokens = SIZE_MAX / 2 + 1;
  const size_t heads = 2;
  const size_t channels = 2;

  ASSERT_EQ(xnn_status_out_of_memory,
            xnn_reshape_rope_nthc_f32(rope_op, batch_size, tokens, heads,
                                      channels, /*threadpool=*/nullptr));
}
