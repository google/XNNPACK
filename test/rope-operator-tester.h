// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <fp16/fp16.h>

#include <xnnpack.h>
#include <xnnpack/cache.h>


class RoPEOperatorTester {
 public:
  enum class WeightsType {
    Default,
    FP32,
  };

  inline RoPEOperatorTester& channels(size_t channels) {
    assert(channels >= 1);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline RoPEOperatorTester& heads(size_t heads) {
    assert(heads >= 1);
    this->heads_ = heads;
    return *this;
  }

  inline size_t heads() const {
    return this->heads_;
  }

  inline RoPEOperatorTester& tokens(size_t tokens) {
    assert(tokens >= 1);
    this->tokens_ = tokens;
    return *this;
  }

  inline size_t tokens() const {
    return this->tokens_;
  }

  inline RoPEOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size >= 1);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline RoPEOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestF16() const {
    ASSERT_EQ(channels() % 2, 0);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32rdist(1.0f, 10.0f);
    std::uniform_real_distribution<float> f32idist(0.01f, 0.1f);

    std::vector<uint16_t> input(XNN_EXTRA_BYTES / sizeof(uint16_t) +
      batch_size() * tokens() * heads() * channels());
    std::vector<uint16_t> weights(XNN_EXTRA_BYTES / sizeof(uint16_t) + tokens() * channels());
    std::vector<uint16_t> output(batch_size() * tokens() * heads() * channels());
    std::vector<float> output_ref(batch_size() * tokens() * heads() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      for (size_t n = 0; n < batch_size(); n++) {
        for (size_t t = 0; t < tokens(); t++) {
          for (size_t h = 0; h < heads(); h++) {
            std::generate_n(input.begin() + ((n * tokens() + t) * heads() + h) * channels(),
                            channels() / 2,
                            [&]() { return fp16_ieee_from_fp32_value(f32rdist(rng)); });
            std::generate_n(input.begin() + (((n * tokens() + t) * heads() + h) * channels() + channels() / 2),
                            channels() / 2,
                            [&]() { return fp16_ieee_from_fp32_value(f32idist(rng)); });
          }
        }
      }
      for (size_t t = 0; t < tokens(); t++) {
        std::generate_n(weights.begin() + t * channels(),
                        channels() / 2,
                        [&]() { return fp16_ieee_from_fp32_value(f32rdist(rng)); });
        std::generate_n(weights.begin() + (t * channels() + channels() / 2),
                        channels() / 2,
                        [&]() { return fp16_ieee_from_fp32_value(f32idist(rng)); });
      }
      std::fill(output.begin(), output.end(), UINT16_C(0xDEAD));
      std::fill(output_ref.begin(), output_ref.end(), std::nan(""));

      // Compute reference results
      for (size_t n = 0; n < batch_size(); n++) {
        for (size_t t = 0; t < tokens(); t++) {
          for (size_t h = 0; h < heads(); h++) {
            for (size_t c = 0; c < channels() / 2; c++) {
              float input_i = fp16_ieee_to_fp32_value(input[((n * tokens() + t) * heads() + h) * channels() + c]);
              float weights_i = fp16_ieee_to_fp32_value(weights[t * channels() + c]);
              float input_n = fp16_ieee_to_fp32_value(input[((n * tokens() + t) * heads() + h) * channels() + (c + channels() / 2)]);
              float weights_n = fp16_ieee_to_fp32_value(weights[t * channels() + (c + channels() / 2)]);
              output_ref[((n * tokens() + t) * heads() + h) * channels() + c] = input_i * weights_i - input_n * weights_n;
              output_ref[((n * tokens() + t) * heads() + h) * channels() + (c + channels() / 2)] = input_i * weights_n + input_n * weights_i;
            }
          }
        }
      }

      // Create, setup, run, and destroy RoPE operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t rope_op = nullptr;

      const xnn_status status = xnn_create_rope_nthc_f16(
        /*max_tokens=*/tokens(), /*flags=*/0, &rope_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, rope_op);

      // Smart pointer to automatically delete rope_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_rope_op(rope_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_rope_nthc_f16(
          rope_op,
          batch_size(), tokens(), heads(), channels(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_rope_nthc_f16(
          rope_op,
          input.data(), weights.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(rope_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t n = 0; n < batch_size(); n++) {
        for (size_t t = 0; t < tokens(); t++) {
          for (size_t h = 0; h < heads(); h++) {
            for (size_t c = 0; c < channels(); c++) {
              const float tolerance = std::abs(output_ref[((n * tokens() + t) * heads() + h) * channels() + c]) * 1.0e-2f;
              ASSERT_NEAR(output_ref[((n * tokens() + t) * heads() + h) * channels() + c],
                          fp16_ieee_to_fp32_value(output[((n * tokens() + t) * heads() + h) * channels() + c]),
                          tolerance)
                  << "batch " << n << " / " << batch_size()
                  << ", token " << t << " / " << tokens()
                  << ", head " << h << " / " << heads()
                  << ", channel " << c << " / " << channels();
            }
          }
        }
      }
    }
  }

  void TestF32() const {
    ASSERT_EQ(channels() % 2, 0);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32rdist(1.0f, 10.0f);
    std::uniform_real_distribution<float> f32idist(0.01f, 0.1f);

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      batch_size() * tokens() * heads() * channels());
    std::vector<float> weights(XNN_EXTRA_BYTES / sizeof(float) + tokens() * channels());
    std::vector<float> output(batch_size() * tokens() * heads() * channels());
    std::vector<double> output_ref(batch_size() * tokens() * heads() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      for (size_t n = 0; n < batch_size(); n++) {
        for (size_t t = 0; t < tokens(); t++) {
          for (size_t h = 0; h < heads(); h++) {
            std::generate_n(input.begin() + ((n * tokens() + t) * heads() + h) * channels(),
                            channels() / 2,
                            [&]() { return f32rdist(rng); });
            std::generate_n(input.begin() + (((n * tokens() + t) * heads() + h) * channels() + channels() / 2),
                            channels() / 2,
                            [&]() { return f32idist(rng); });
          }
        }
      }
      for (size_t t = 0; t < tokens(); t++) {
        std::generate_n(weights.begin() + t * channels(),
                        channels() / 2,
                        [&]() { return f32rdist(rng); });
        std::generate_n(weights.begin() + (t * channels() + channels() / 2),
                        channels() / 2,
                        [&]() { return f32idist(rng); });
      }
      std::fill(output.begin(), output.end(), std::nanf(""));
      std::fill(output_ref.begin(), output_ref.end(), std::nan(""));

      // Compute reference results
      for (size_t n = 0; n < batch_size(); n++) {
        for (size_t t = 0; t < tokens(); t++) {
          for (size_t h = 0; h < heads(); h++) {
            for (size_t c = 0; c < channels() / 2; c++) {
              output_ref[((n * tokens() + t) * heads() + h) * channels() + c] =
                double(input[((n * tokens() + t) * heads() + h) * channels() + c]) *
                  double(weights[t * channels() + c]) -
                double(input[((n * tokens() + t) * heads() + h) * channels() + (c + channels() / 2)]) *
                  double(weights[t * channels() + (c + channels() / 2)]);
              output_ref[((n * tokens() + t) * heads() + h) * channels() + (c + channels() / 2)] =
                double(input[((n * tokens() + t) * heads() + h) * channels() + c]) *
                  double(weights[t * channels() + (c + channels() / 2)]) +
                double(input[((n * tokens() + t) * heads() + h) * channels() + (c + channels() / 2)]) *
                  double(weights[t * channels() + c]);
            }
          }
        }
      }

      // Create, setup, run, and destroy RoPE operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t rope_op = nullptr;

      const xnn_status status = xnn_create_rope_nthc_f32(
        /*max_tokens=*/tokens(), /*flags=*/0, &rope_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, rope_op);

      // Smart pointer to automatically delete rope_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_rope_op(rope_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_rope_nthc_f32(
          rope_op,
          batch_size(), tokens(), heads(), channels(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_rope_nthc_f32(
          rope_op,
          input.data(), weights.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(rope_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t n = 0; n < batch_size(); n++) {
        for (size_t t = 0; t < tokens(); t++) {
          for (size_t h = 0; h < heads(); h++) {
            for (size_t c = 0; c < channels(); c++) {
              ASSERT_NEAR(output_ref[((n * tokens() + t) * heads() + h) * channels() + c],
                          output[((n * tokens() + t) * heads() + h) * channels() + c],
                          1.0e-4 * std::abs(output_ref[((n * tokens() + t) * heads() + h) * channels() + c]))
                  << "batch " << n << " / " << batch_size()
                  << ", token " << t << " / " << tokens()
                  << ", head " << h << " / " << heads()
                  << ", channel " << c << " / " << channels();
            }
          }
        }
      }
    }
  }

 private:
  size_t channels_{1};
  size_t heads_{1};
  size_t tokens_{1};
  size_t batch_size_{1};
  size_t iterations_{3};
};
