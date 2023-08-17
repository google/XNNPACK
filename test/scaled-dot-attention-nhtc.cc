// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>

#include <gtest/gtest.h>

#include "attention-operator-tester.h"


TEST(ATTENTION_NHTC_F32, unit_batch) {
  AttentionOperatorTester()
      .batch_size(1)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, multi_head) {
  AttentionOperatorTester()
      .query_heads(13)
      .key_value_heads(13)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, multi_query) {
  AttentionOperatorTester()
      .query_heads(13)
      .key_value_heads(1)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, batch_size) {
  AttentionOperatorTester()
      .batch_size(13)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, batch_size_multi_head) {
  AttentionOperatorTester()
      .batch_size(13)
      .query_heads(17)
      .key_value_heads(17)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, batch_size_multi_query) {
  AttentionOperatorTester()
      .batch_size(13)
      .query_heads(17)
      .key_value_heads(1)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, large_channels) {
  AttentionOperatorTester()
      .batch_size(1)
      .query_tokens(41)
      .query_key_channels(1543)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, different_channels) {
  AttentionOperatorTester()
      .batch_size(3)
      .query_heads(5)
      .key_value_heads(5)
      .query_tokens(41)
      .key_value_tokens(41)
      .query_key_channels(137)
      .value_channels(61)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, self_attention_with_cap) {
  AttentionOperatorTester()
      .cap_tanh(20.0f)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, multi_head_self_attention_with_cap) {
  AttentionOperatorTester()
      .query_heads(17)
      .key_value_heads(17)
      .cap_tanh(20.0f)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, multi_query_self_attention_with_cap) {
  AttentionOperatorTester()
      .query_heads(17)
      .key_value_heads(1)
      .cap_tanh(20.0f)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, multi_head_cross_attention_key_value_tokens_lt_query_tokens) {
  AttentionOperatorTester()
      .query_heads(17)
      .key_value_heads(17)
      .query_tokens(41)
      .key_value_tokens(29)
      .query_key_channels(137)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, multi_head_cross_attention_key_value_tokens_gt_query_tokens) {
  AttentionOperatorTester()
      .query_heads(17)
      .key_value_heads(17)
      .query_tokens(29)
      .key_value_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, multi_query_cross_attention_key_value_tokens_lt_query_tokens) {
  AttentionOperatorTester()
      .query_heads(17)
      .key_value_heads(1)
      .query_tokens(41)
      .key_value_tokens(29)
      .query_key_channels(137)
      .TestF32();
}

TEST(ATTENTION_NHTC_F32, multi_query_cross_attention_key_value_tokens_gt_query_tokens) {
  AttentionOperatorTester()
      .query_heads(17)
      .key_value_heads(1)
      .query_tokens(29)
      .key_value_tokens(41)
      .query_key_channels(137)
      .TestF32();
}
