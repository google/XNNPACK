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

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>

#include <gtest/gtest.h>


class AttentionOperatorTester {
 public:
  inline AttentionOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline AttentionOperatorTester& cap_tanh(float cap) {
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

  inline AttentionOperatorTester& query_tokens(size_t query_tokens) {
    this->query_tokens_ = query_tokens;
    return *this;
  }

  inline size_t query_tokens() const {
    return this->query_tokens_;
  }

  inline AttentionOperatorTester& key_value_tokens(size_t key_value_tokens) {
    this->key_value_tokens_ = key_value_tokens;
    return *this;
  }

  inline size_t key_value_tokens() const {
    if (this->key_value_tokens_ == 0) return query_tokens();
    return this->key_value_tokens_;
  }

  inline AttentionOperatorTester& channels(size_t channels) {
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline AttentionOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> scaledist(0.2f, 2.0f);

    std::vector<float> query(XNN_EXTRA_BYTES / sizeof(float) + batch_size() * query_tokens() * channels());
    std::vector<float> key(XNN_EXTRA_BYTES / sizeof(float) + batch_size() * key_value_tokens() * channels());
    std::vector<float> value(XNN_EXTRA_BYTES / sizeof(float) + batch_size() * key_value_tokens() * channels());
    std::vector<float> scale(XNN_EXTRA_BYTES / sizeof(float) + channels());
    std::vector<float> mask(XNN_EXTRA_BYTES / sizeof(float) + query_tokens() * key_value_tokens());
    std::vector<float> output(batch_size() * query_tokens() * channels());
    std::vector<float> output_ref(batch_size() * query_tokens() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(query.begin(), query.end(), [&]() { return f32dist(rng); });
      // Use a different distribution to avoid divide by 0.
      std::generate(scale.begin(), scale.end(), [&]() { return scaledist(rng); });
      std::generate(key.begin(), key.end(), [&]() { return f32dist(rng); });
      std::generate(value.begin(), value.end(), [&]() { return f32dist(rng); });
      std::generate(mask.begin(), mask.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      for (size_t b = 0; b < batch_size(); b++) {
        // Compute reference results.
        std::vector<float> q_scaled(query_tokens() * channels());
        for (size_t n = 0; n < query_tokens(); n++) {
          for (size_t k = 0; k < channels(); k++) {
            q_scaled[n * channels() + k] =
                query[b * query_tokens() * channels() + n * channels() + k] * scale[k];
          }
        }

        std::vector<float> logits(query_tokens() * key_value_tokens(), 0);
        for (size_t n_0 = 0; n_0 < query_tokens(); n_0++) {
          for (size_t n_1 = 0; n_1 < key_value_tokens(); n_1++) {
            for (size_t ki = 0; ki < channels(); ki++) {
              logits[n_0 * key_value_tokens() + n_1] +=
                  q_scaled[n_0 * channels() + ki] *
                  key[b * key_value_tokens() * channels() + n_1 * channels() + ki];
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
            for (size_t di = 0; di < channels(); di++) {
              output_ref[b * query_tokens() * channels() + ni * channels() + di] +=
                  weights[ni * key_value_tokens() + nj] *
                  value[b * key_value_tokens() * channels() + nj * channels() + di];
            }
          }
        }
      }

      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t attention_op = nullptr;
      xnn_attention_logits_cap_tanh_params cap_tanh_params = {cap_value()};
      const xnn_status status = xnn_create_scaled_dot_attention_ntc_f32(
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
                xnn_reshape_scaled_dot_attention_ntc_f32(
                  attention_op,
                  batch_size(), query_tokens(), key_value_tokens(), channels(),
                  &workspace_size, &workspace_alignment,
                  /*threadpool=*/nullptr));

      ASSERT_NE(workspace_size, 0);
      ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size, 0);

      ASSERT_EQ(xnn_status_success,
                xnn_setup_scaled_dot_attention_ntc_f32(
                  attention_op,
                  workspace.data(), query.data(), key.data(), value.data(),
                  scale.data(), mask.data(), output.data()));

      ASSERT_EQ(xnn_status_success, xnn_run_operator(attention_op, /*threadpool=*/nullptr));

      for (size_t b = 0; b < batch_size(); b++) {
        for (size_t i = 0; i < query_tokens(); i++) {
          for (size_t j = 0; j < channels(); j++) {
            EXPECT_NEAR(output_ref[b * query_tokens() * channels() + i * channels() + j],
                        output[b * query_tokens() * channels() + i * channels() + j],
                        1e-4)
                << " batch : " << b << " / "  << batch_size()
                << " token : " << i << " / " << query_tokens()
                << " channel : " << j << " / " << channels();
          }
        }
      }
    }
  }

 private:
  xnn_attention_logits_cap_type cap_type_ = xnn_attention_logits_cap_type_none;
  float cap_value_{0.0f};
  size_t batch_size_{1};
  size_t channels_{1};
  size_t query_tokens_{1};
  size_t key_value_tokens_{0};
  size_t iterations_{1};
};
