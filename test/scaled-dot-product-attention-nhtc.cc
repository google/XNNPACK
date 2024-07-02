// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <gtest/gtest.h>
#include "scaled-dot-product-attention-operator-tester.h"


TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F16, unit_batch) {
  ScaledDotProductAttentionOperatorTester()
      .batch_size(1)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF16();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F16, multi_head) {
  ScaledDotProductAttentionOperatorTester()
      .query_heads(13)
      .key_value_heads(13)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF16();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F16, multi_query) {
  ScaledDotProductAttentionOperatorTester()
      .query_heads(13)
      .key_value_heads(1)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF16();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F16, batch_size) {
  ScaledDotProductAttentionOperatorTester()
      .batch_size(13)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF16();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F16, batch_size_multi_head) {
  ScaledDotProductAttentionOperatorTester()
      .batch_size(13)
      .query_heads(17)
      .key_value_heads(17)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF16();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F16, batch_size_multi_query) {
  ScaledDotProductAttentionOperatorTester()
      .batch_size(13)
      .query_heads(17)
      .key_value_heads(1)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF16();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F16, different_channels) {
  ScaledDotProductAttentionOperatorTester()
      .batch_size(3)
      .query_heads(5)
      .key_value_heads(5)
      .query_tokens(41)
      .key_value_tokens(41)
      .query_key_channels(137)
      .value_channels(61)
      .TestF16();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F16, self_attention_with_cap) {
  ScaledDotProductAttentionOperatorTester()
      .cap_tanh(20.0f)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF16();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F16, multi_head_self_attention_with_cap) {
  ScaledDotProductAttentionOperatorTester()
      .query_heads(17)
      .key_value_heads(17)
      .cap_tanh(20.0f)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF16();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F16, multi_query_self_attention_with_cap) {
  ScaledDotProductAttentionOperatorTester()
      .query_heads(17)
      .key_value_heads(1)
      .cap_tanh(20.0f)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF16();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F16, multi_head_cross_attention_key_value_tokens_lt_query_tokens) {
  ScaledDotProductAttentionOperatorTester()
      .query_heads(17)
      .key_value_heads(17)
      .query_tokens(41)
      .key_value_tokens(29)
      .query_key_channels(137)
      .TestF16();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F16, multi_head_cross_attention_key_value_tokens_gt_query_tokens) {
  ScaledDotProductAttentionOperatorTester()
      .query_heads(17)
      .key_value_heads(17)
      .query_tokens(29)
      .key_value_tokens(41)
      .query_key_channels(137)
      .TestF16();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F16, multi_query_cross_attention_key_value_tokens_lt_query_tokens) {
  ScaledDotProductAttentionOperatorTester()
      .query_heads(17)
      .key_value_heads(1)
      .query_tokens(41)
      .key_value_tokens(29)
      .query_key_channels(137)
      .TestF16();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F16, multi_query_cross_attention_key_value_tokens_gt_query_tokens) {
  ScaledDotProductAttentionOperatorTester()
      .query_heads(17)
      .key_value_heads(1)
      .query_tokens(29)
      .key_value_tokens(41)
      .query_key_channels(137)
      .TestF16();
}

// Small parallelization terms to test the case where we size workspace using batch size.
TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F16, multi_head_cross_attention_multithreaded_small_batch) {
  ScaledDotProductAttentionOperatorTester()
      .batch_size(1)
      .query_heads(1)
      .key_value_heads(1)
      .query_tokens(1)
      .multithreaded(true)
      .TestF16();
}

// Large parallelization terms to test the case where we size workspace using number of threads.
TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F16, multi_head_cross_attention_multithreaded) {
  ScaledDotProductAttentionOperatorTester()
      .batch_size(31)
      .query_heads(17)
      .key_value_heads(17)
      .query_tokens(29)
      .key_value_tokens(41)
      .query_key_channels(137)
      .value_channels(61)
      .multithreaded(true)
      .TestF16();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F32, unit_batch) {
  ScaledDotProductAttentionOperatorTester()
      .batch_size(1)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F32, multi_head) {
  ScaledDotProductAttentionOperatorTester()
      .query_heads(13)
      .key_value_heads(13)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F32, multi_query) {
  ScaledDotProductAttentionOperatorTester()
      .query_heads(13)
      .key_value_heads(1)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F32, batch_size) {
  ScaledDotProductAttentionOperatorTester()
      .batch_size(13)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F32, batch_size_multi_head) {
  ScaledDotProductAttentionOperatorTester()
      .batch_size(13)
      .query_heads(17)
      .key_value_heads(17)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F32, batch_size_multi_query) {
  ScaledDotProductAttentionOperatorTester()
      .batch_size(13)
      .query_heads(17)
      .key_value_heads(1)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F32, large_channels) {
  ScaledDotProductAttentionOperatorTester()
      .batch_size(1)
      .query_tokens(41)
      .query_key_channels(1543)
      .TestF32();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F32, different_channels) {
  ScaledDotProductAttentionOperatorTester()
      .batch_size(3)
      .query_heads(5)
      .key_value_heads(5)
      .query_tokens(41)
      .key_value_tokens(41)
      .query_key_channels(137)
      .value_channels(61)
      .TestF32();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F32, self_attention_with_cap) {
  ScaledDotProductAttentionOperatorTester()
      .cap_tanh(20.0f)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F32, multi_head_self_attention_with_cap) {
  ScaledDotProductAttentionOperatorTester()
      .query_heads(17)
      .key_value_heads(17)
      .cap_tanh(20.0f)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F32, multi_query_self_attention_with_cap) {
  ScaledDotProductAttentionOperatorTester()
      .query_heads(17)
      .key_value_heads(1)
      .cap_tanh(20.0f)
      .query_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F32, multi_head_cross_attention_key_value_tokens_lt_query_tokens) {
  ScaledDotProductAttentionOperatorTester()
      .query_heads(17)
      .key_value_heads(17)
      .query_tokens(41)
      .key_value_tokens(29)
      .query_key_channels(137)
      .TestF32();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F32, multi_head_cross_attention_key_value_tokens_gt_query_tokens) {
  ScaledDotProductAttentionOperatorTester()
      .query_heads(17)
      .key_value_heads(17)
      .query_tokens(29)
      .key_value_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F32, multi_query_cross_attention_key_value_tokens_lt_query_tokens) {
  ScaledDotProductAttentionOperatorTester()
      .query_heads(17)
      .key_value_heads(1)
      .query_tokens(41)
      .key_value_tokens(29)
      .query_key_channels(137)
      .TestF32();
}

TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F32, multi_query_cross_attention_key_value_tokens_gt_query_tokens) {
  ScaledDotProductAttentionOperatorTester()
      .query_heads(17)
      .key_value_heads(1)
      .query_tokens(29)
      .key_value_tokens(41)
      .query_key_channels(137)
      .TestF32();
}

// Small parallelization terms to test the case where we size workspace using batch size.
TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F32, multi_head_cross_attention_multithreaded_small_batch) {
  ScaledDotProductAttentionOperatorTester()
      .batch_size(1)
      .query_heads(1)
      .key_value_heads(1)
      .query_tokens(1)
      .multithreaded(true)
      .TestF32();
}

// Large parallelization terms to test the case where we size workspace using number of threads.
TEST(SCALED_DOT_PRODUCT_ATTENTION_NHTC_F32, multi_head_cross_attention_multithreaded) {
  ScaledDotProductAttentionOperatorTester()
      .batch_size(31)
      .query_heads(17)
      .key_value_heads(17)
      .query_tokens(29)
      .key_value_tokens(41)
      .query_key_channels(137)
      .value_channels(61)
      .multithreaded(true)
      .TestF32();
}
