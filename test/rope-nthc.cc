// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include "rope-operator-tester.h"


TEST(ROPE_NTHC_F16, two_channels) {
  RoPEOperatorTester()
    .batch_size(1)
    .heads(1)
    .tokens(1)
    .channels(2)
    .TestF16();
}

TEST(ROPE_NTHC_F16, multiple_channels) {
  RoPEOperatorTester()
    .batch_size(1)
    .heads(1)
    .tokens(1)
    .channels(42)
    .TestF16();
}

TEST(ROPE_NTHC_F16, multiple_tokens) {
  RoPEOperatorTester()
    .batch_size(1)
    .heads(1)
    .tokens(11)
    .channels(42)
    .TestF16();
}

TEST(ROPE_NTHC_F16, multiple_heads) {
  RoPEOperatorTester()
    .batch_size(1)
    .heads(7)
    .tokens(11)
    .channels(42)
    .TestF16();
}

TEST(ROPE_NTHC_F16, nonunit_batch) {
  RoPEOperatorTester()
    .batch_size(3)
    .heads(7)
    .tokens(11)
    .channels(42)
    .TestF16();
}

TEST(ROPE_NTHC_F32, two_channels) {
  RoPEOperatorTester()
    .batch_size(1)
    .heads(1)
    .tokens(1)
    .channels(2)
    .TestF32();
}

TEST(ROPE_NTHC_F32, multiple_channels) {
  RoPEOperatorTester()
    .batch_size(1)
    .heads(1)
    .tokens(1)
    .channels(42)
    .TestF32();
}

TEST(ROPE_NTHC_F32, multiple_tokens) {
  RoPEOperatorTester()
    .batch_size(1)
    .heads(1)
    .tokens(11)
    .channels(42)
    .TestF32();
}

TEST(ROPE_NTHC_F32, multiple_heads) {
  RoPEOperatorTester()
    .batch_size(1)
    .heads(7)
    .tokens(11)
    .channels(42)
    .TestF32();
}

TEST(ROPE_NTHC_F32, nonunit_batch) {
  RoPEOperatorTester()
    .batch_size(3)
    .heads(7)
    .tokens(11)
    .channels(42)
    .TestF32();
}
