// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <fp16/fp16.h>
#include <pthreadpool.h>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>

#include <gtest/gtest.h>


class ScaledDotProductAttentionOperatorTester {
 public:
  inline ScaledDotProductAttentionOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline ScaledDotProductAttentionOperatorTester& query_heads(size_t query_heads) {
    assert(query_heads != 0);
    this->query_heads_ = query_heads;
    return *this;
  }

  inline size_t query_heads() const {
    return this->query_heads_;
  }

  inline ScaledDotProductAttentionOperatorTester& key_value_heads(size_t key_value_heads) {
    assert(key_value_heads == 1 || key_value_heads == query_heads());
    this->key_value_heads_ = key_value_heads;
    return *this;
  }

  inline size_t key_value_heads() const {
    return this->key_value_heads_;
  }

  inline ScaledDotProductAttentionOperatorTester& cap_tanh(float cap) {
    this->cap_type_ = xnn_attention_logits_cap_type_tanh;
    this->cap_value_ = cap;
    return *this;
  }

  inline xnn_attention_logits_cap_type cap_type() const {
    return this->cap_type_;
  }

  inline float cap_value() const {
    return this->cap_value_;
  }

  inline ScaledDotProductAttentionOperatorTester& query_tokens(size_t query_tokens) {
    this->query_tokens_ = query_tokens;
    return *this;
  }

  inline size_t query_tokens() const {
    return this->query_tokens_;
  }

  inline ScaledDotProductAttentionOperatorTester& key_value_tokens(size_t key_value_tokens) {
    this->key_value_tokens_ = key_value_tokens;
    return *this;
  }

  inline size_t key_value_tokens() const {
    if (this->key_value_tokens_ == 0) return query_tokens();
    return this->key_value_tokens_;
  }

  inline ScaledDotProductAttentionOperatorTester& query_key_channels(size_t query_key_channels) {
    this->query_key_channels_ = query_key_channels;
    return *this;
  }

  inline size_t query_key_channels() const {
    return this->query_key_channels_;
  }

  inline ScaledDotProductAttentionOperatorTester& value_channels(size_t value_channels) {
    this->value_channels_ = value_channels;
    return *this;
  }

  inline size_t value_channels() const {
    return this->value_channels_;
  }

  inline ScaledDotProductAttentionOperatorTester& multithreaded(bool multithreaded) {
    this->multithreaded_ = multithreaded;
    return *this;
  }

  inline bool multithreaded() const {
    return this->multithreaded_;
  }

  inline size_t num_threads() const {
    return multithreaded() ? 5 : 1;
  }

  inline ScaledDotProductAttentionOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestF16() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(0.1, 1.0f);
    // Use a different scale distributon to mitigate precision issues.
    // In tests, channels are ~100, so scale is ~0.1.
    const float dk_scale = 1.0f / std::sqrt(static_cast<float>(query_key_channels()));
    std::uniform_real_distribution<float> scaledist(std::min(0.01f, dk_scale), std::max(0.01f, dk_scale));

    std::vector<uint16_t> query(XNN_EXTRA_BYTES / sizeof(uint16_t) + batch_size() * query_heads() * query_tokens() * query_key_channels());
    std::vector<uint16_t> key(XNN_EXTRA_BYTES / sizeof(uint16_t) + batch_size() * key_value_heads() * key_value_tokens() * query_key_channels());
    std::vector<uint16_t> value(XNN_EXTRA_BYTES / sizeof(uint16_t) + batch_size() * key_value_heads() * key_value_tokens() * value_channels());
    std::vector<uint16_t> scale(XNN_EXTRA_BYTES / sizeof(uint16_t) + query_key_channels());
    std::vector<uint16_t> mask(XNN_EXTRA_BYTES / sizeof(uint16_t) + query_tokens() * key_value_tokens());
    std::vector<uint16_t> output(batch_size() * query_heads() * query_tokens() * value_channels());
    std::vector<float> output_ref(batch_size() * query_heads() * query_tokens() * value_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(query.begin(), query.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      // Use a different distribution to avoid divide by 0.
      std::generate(scale.begin(), scale.end(), [&]() { return fp16_ieee_from_fp32_value(scaledist(rng)); });
      std::generate(key.begin(), key.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(value.begin(), value.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(mask.begin(), mask.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      const size_t query_batch_stride = query_heads() *  query_tokens() * query_key_channels();
      const size_t query_head_stride = query_tokens() * query_key_channels();
      const size_t output_batch_stride = query_heads() *  query_tokens() * value_channels();
      const size_t output_head_stride = query_tokens() * value_channels();
      const size_t key_batch_stride = key_value_heads() *  key_value_tokens() * query_key_channels();
      const size_t value_batch_stride = key_value_heads() *  key_value_tokens() * value_channels();
      // For multi-query, key/value only has single head, so don't advance along head dimension.
      const size_t key_head_stride = key_value_heads() == 1 ? 0 : key_value_tokens() * query_key_channels();
      const size_t value_head_stride = key_value_heads() == 1 ? 0 : key_value_tokens() * value_channels();

      for (size_t b = 0; b < batch_size(); b++) {
        for (size_t h = 0; h < query_heads(); h++) {
          // Compute reference results.
          std::vector<float> q_scaled(query_tokens() * query_key_channels());
          for (size_t n = 0; n < query_tokens(); n++) {
            for (size_t k = 0; k < query_key_channels(); k++) {
              q_scaled[n * query_key_channels() + k] =
                fp16_ieee_to_fp32_value(query[b * query_batch_stride + h * query_head_stride + n * query_key_channels() + k]) *
                fp16_ieee_to_fp32_value(scale[k]);
            }
          }

          std::vector<float> logits(query_tokens() * key_value_tokens(), 0.0f);
          for (size_t n_0 = 0; n_0 < query_tokens(); n_0++) {
            for (size_t n_1 = 0; n_1 < key_value_tokens(); n_1++) {
              for (size_t ki = 0; ki < query_key_channels(); ki++) {
                logits[n_0 * key_value_tokens() + n_1] +=
                  (q_scaled[n_0 * query_key_channels() + ki]) *
                  fp16_ieee_to_fp32_value(key[b * key_batch_stride + h * key_head_stride + n_1 * query_key_channels() + ki]);
              }
              if (cap_type() == xnn_attention_logits_cap_type_tanh) {
                // Cap and tanh.
                logits[n_0 * key_value_tokens() + n_1] =
                  std::tanh((logits[n_0 * key_value_tokens() + n_1]) / cap_value()) * cap_value();
              }
              // Mask.
              logits[n_0 * key_value_tokens() + n_1] +=
                fp16_ieee_to_fp32_value(mask[n_0 * key_value_tokens() + n_1]);
            }
          }

          std::vector<float> weights(query_tokens() * key_value_tokens(), 0.0f);

          for (size_t i = 0; i < query_tokens(); i++) {
            // Online softmax per row.
            float mv = -std::numeric_limits<double>::infinity();
            float dv = 0;
            for (size_t j = 0; j < key_value_tokens(); j++) {
              float prev_m = mv;
              mv = std::max(prev_m, logits[i * key_value_tokens() + j]);
              dv = dv * exp(prev_m - mv) + exp(logits[i * key_value_tokens() + j] - mv);
            }
            for (size_t j = 0; j < key_value_tokens(); j++) {
              weights[i * key_value_tokens() + j] =
                  exp(logits[i * key_value_tokens() + j] - mv)/ dv;
            }
          }

          // Output = Weights * Value
          for (size_t ni = 0; ni < query_tokens(); ni++) {
            for (size_t nj = 0; nj < key_value_tokens(); nj++) {
              for (size_t di = 0; di < value_channels(); di++) {
                output_ref[b * output_batch_stride + h * output_head_stride + ni * value_channels() + di] +=
                    weights[ni * key_value_tokens() + nj] *
                    fp16_ieee_to_fp32_value(value[b * value_batch_stride + h * value_head_stride + nj * value_channels() + di]);
              }
            }
          }
        }
      }

      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t attention_op = nullptr;
      xnn_attention_logits_cap_tanh_params cap_tanh_params = {cap_value()};
      const xnn_status status = xnn_create_scaled_dot_product_attention_nhtc_f16(
          cap_type(),
          &cap_tanh_params,
          /*flags=*/0,
          &attention_op);

      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(attention_op, nullptr);

      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_attention_op(attention_op, xnn_delete_operator);

      size_t workspace_size = 0;
      size_t workspace_alignment = 0;
      ASSERT_EQ(xnn_status_success,
                xnn_reshape_scaled_dot_product_attention_nhtc_f16(
                  attention_op,
                  batch_size(), query_heads(), query_tokens(),
                  key_value_heads(), key_value_tokens(),
                  query_key_channels(), value_channels(),
                  &workspace_size, &workspace_alignment,
                  auto_threadpool.get()));

      ASSERT_NE(workspace_size, 0);
      ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size, 0);

      ASSERT_EQ(xnn_status_success,
                xnn_setup_scaled_dot_product_attention_nhtc_f16(
                  attention_op,
                  workspace.data(), query.data(), key.data(), value.data(),
                  scale.data(), mask.data(), output.data()));

      ASSERT_EQ(xnn_status_success, xnn_run_operator(attention_op, auto_threadpool.get()));

      for (size_t b = 0; b < batch_size(); b++) {
        for (size_t h = 0; h < query_heads(); h++) {
          for (size_t i = 0; i < query_tokens(); i++) {
            for (size_t j = 0; j < value_channels(); j++) {
              EXPECT_NEAR(output_ref[(b * query_heads() + h) * query_tokens() * value_channels() + i * value_channels() + j],
                          fp16_ieee_to_fp32_value(output[(b * query_heads() + h) * query_tokens() * value_channels() + i * value_channels() + j]),
                          1e-2)
                  << " batch : " << b << " / "  << batch_size()
                  << " head : " << h << " / "  << query_heads()
                  << " token : " << i << " / " << query_tokens()
                  << " channel : " << j << " / " << value_channels();
            }
          }
        }
      }
    }
  }

  void TestF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> scaledist(0.2f, 2.0f);

    std::vector<float> query(XNN_EXTRA_BYTES / sizeof(float) + batch_size() * query_heads() * query_tokens() * query_key_channels());
    std::vector<float> key(XNN_EXTRA_BYTES / sizeof(float) + batch_size() * key_value_heads() * key_value_tokens() * query_key_channels());
    std::vector<float> value(XNN_EXTRA_BYTES / sizeof(float) + batch_size() * key_value_heads() * key_value_tokens() * value_channels());
    std::vector<float> scale(XNN_EXTRA_BYTES / sizeof(float) + query_key_channels());
    std::vector<float> mask(XNN_EXTRA_BYTES / sizeof(float) + query_tokens() * key_value_tokens());
    std::vector<float> output(batch_size() * query_heads() * query_tokens() * value_channels());
    std::vector<float> output_ref(batch_size() * query_heads() * query_tokens() * value_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(query.begin(), query.end(), [&]() { return f32dist(rng); });
      // Use a different distribution to avoid divide by 0.
      std::generate(scale.begin(), scale.end(), [&]() { return scaledist(rng); });
      std::generate(key.begin(), key.end(), [&]() { return f32dist(rng); });
      std::generate(value.begin(), value.end(), [&]() { return f32dist(rng); });
      std::generate(mask.begin(), mask.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      const size_t query_batch_stride = query_heads() *  query_tokens() * query_key_channels();
      const size_t query_head_stride = query_tokens() * query_key_channels();
      const size_t output_batch_stride = query_heads() *  query_tokens() * value_channels();
      const size_t output_head_stride = query_tokens() * value_channels();
      const size_t key_batch_stride = key_value_heads() *  key_value_tokens() * query_key_channels();
      const size_t value_batch_stride = key_value_heads() *  key_value_tokens() * value_channels();
      // For multi-query, key/value only has single head, so don't advance along head dimension.
      const size_t key_head_stride = key_value_heads() == 1 ? 0 : key_value_tokens() * query_key_channels();
      const size_t value_head_stride = key_value_heads() == 1 ? 0 : key_value_tokens() * value_channels();

      for (size_t b = 0; b < batch_size(); b++) {
        for (size_t h = 0; h < query_heads(); h++) {
          // Compute reference results.
          std::vector<float> q_scaled(query_tokens() * query_key_channels());
          for (size_t n = 0; n < query_tokens(); n++) {
            for (size_t k = 0; k < query_key_channels(); k++) {
              q_scaled[n * query_key_channels() + k] =
                  query[b * query_batch_stride + h * query_head_stride + n * query_key_channels() + k] * scale[k];
            }
          }

          std::vector<float> logits(query_tokens() * key_value_tokens(), 0);
          for (size_t n_0 = 0; n_0 < query_tokens(); n_0++) {
            for (size_t n_1 = 0; n_1 < key_value_tokens(); n_1++) {
              for (size_t ki = 0; ki < query_key_channels(); ki++) {
                logits[n_0 * key_value_tokens() + n_1] +=
                    q_scaled[n_0 * query_key_channels() + ki] *
                    key[b * key_batch_stride + h * key_head_stride + n_1 * query_key_channels() + ki];
              }
              if (cap_type() == xnn_attention_logits_cap_type_tanh) {
                // Cap and tanh.
                logits[n_0 * key_value_tokens() + n_1] =
                    std::tanh(logits[n_0 * key_value_tokens() + n_1] / cap_value()) * cap_value();
              }
              // Mask.
              logits[n_0 * key_value_tokens() + n_1] += mask[n_0 * key_value_tokens() + n_1];
            }
          }

          std::vector<float> weights(query_tokens() * key_value_tokens(), 0.0f);

          for (size_t i = 0; i < query_tokens(); i++) {
            // Online softmax per row.
            float mv = -std::numeric_limits<double>::infinity();
            float dv = 0;
            for (size_t j = 0; j < key_value_tokens(); j++) {
              float prev_m = mv;
              mv = std::max(prev_m, logits[i * key_value_tokens() + j]);
              dv = dv * exp(prev_m - mv) + exp(logits[i * key_value_tokens() + j] - mv);
            }
            for (size_t j = 0; j < key_value_tokens(); j++) {
              weights[i * key_value_tokens() + j] = exp(logits[i * key_value_tokens() + j] - mv)/ dv;
            }
          }

          // Output = Weights * Value
          for (size_t ni = 0; ni < query_tokens(); ni++) {
            for (size_t nj = 0; nj < key_value_tokens(); nj++) {
              for (size_t di = 0; di < value_channels(); di++) {
                output_ref[b * output_batch_stride + h * output_head_stride + ni * value_channels() + di] +=
                    weights[ni * key_value_tokens() + nj] *
                    value[b * value_batch_stride + h * value_head_stride + nj * value_channels() + di];
              }
            }
          }
        }
      }

      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t attention_op = nullptr;
      xnn_attention_logits_cap_tanh_params cap_tanh_params = {cap_value()};
      const xnn_status status = xnn_create_scaled_dot_product_attention_nhtc_f32(
          cap_type(),
          &cap_tanh_params,
          /*flags=*/0,
          &attention_op);

      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(attention_op, nullptr);

      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_attention_op(attention_op, xnn_delete_operator);


      size_t workspace_size = 0;
      size_t workspace_alignment = 0;
      ASSERT_EQ(xnn_status_success,
                xnn_reshape_scaled_dot_product_attention_nhtc_f32(
                  attention_op,
                  batch_size(), query_heads(), query_tokens(), key_value_heads(), key_value_tokens(),
                    query_key_channels(), value_channels(),
                  &workspace_size, &workspace_alignment,
                  auto_threadpool.get()));

      ASSERT_NE(workspace_size, 0);
      ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size, 0);

      ASSERT_EQ(xnn_status_success,
                xnn_setup_scaled_dot_product_attention_nhtc_f32(
                  attention_op,
                  workspace.data(), query.data(), key.data(), value.data(),
                  scale.data(), mask.data(), output.data()));

      ASSERT_EQ(xnn_status_success, xnn_run_operator(attention_op, auto_threadpool.get()));

      for (size_t b = 0; b < batch_size(); b++) {
        for (size_t h = 0; h < query_heads(); h++) {
          for (size_t i = 0; i < query_tokens(); i++) {
            for (size_t j = 0; j < value_channels(); j++) {
              EXPECT_NEAR(output_ref[(b * query_heads() + h) * query_tokens() * value_channels() + i * value_channels() + j],
                          output[(b * query_heads() + h) * query_tokens() * value_channels() + i * value_channels() + j],
                          1e-4)
                  << " batch : " << b << " / "  << batch_size()
                  << " head : " << h << " / "  << query_heads()
                  << " token : " << i << " / " << query_tokens()
                  << " channel : " << j << " / " << value_channels();
            }
          }
        }
      }
    }
  }

 private:
  xnn_attention_logits_cap_type cap_type_ = xnn_attention_logits_cap_type_none;
  float cap_value_{0.0f};
  size_t batch_size_{1};
  size_t query_heads_{1};
  size_t key_value_heads_{1};
  size_t query_key_channels_{1};
  size_t value_channels_{1};
  size_t query_tokens_{1};
  size_t key_value_tokens_{0};
  bool multithreaded_{false};
  size_t iterations_{1};
};
