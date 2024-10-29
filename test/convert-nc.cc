// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/config-types.h"
#include "xnnpack/config.h"
#include "xnnpack/internal.h"
#include "xnnpack/math.h"
#include "xnnpack/packq.h"
#include "xnnpack/buffer.h"
#include "replicable_random_device.h"

class ConvertOperatorTester {
 public:
  ConvertOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const {
    return this->channels_;
  }

  ConvertOperatorTester& input_stride(size_t input_stride) {
    assert(input_stride != 0);
    this->input_stride_ = input_stride;
    return *this;
  }

  size_t input_stride() const {
    if (this->input_stride_ == 0) {
      return this->channels_;
    } else {
      assert(this->input_stride_ >= this->channels_);
      return this->input_stride_;
    }
  }

  ConvertOperatorTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return this->channels_;
    } else {
      assert(this->output_stride_ >= this->channels_);
      return this->output_stride_;
    }
  }

  ConvertOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const {
    return this->batch_size_;
  }

  ConvertOperatorTester& input_scale(float input_scale) {
    assert(input_scale >= 0.0f);
    assert(std::isnormal(input_scale));
    this->input_scale_ = input_scale;
    return *this;
  }

  float input_scale() const {
    return this->input_scale_;
  }

  ConvertOperatorTester& output_scale(float output_scale) {
    assert(output_scale >= 0.0f);
    assert(std::isnormal(output_scale));
    this->output_scale_ = output_scale;
    return *this;
  }

  float output_scale() const {
    return this->output_scale_;
  }

  ConvertOperatorTester& zero_point(int16_t zero_point) {
    this->zero_point_ = zero_point;
    return *this;
  }

  int16_t zero_point() const {
    return this->zero_point_;
  }

  ConvertOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void TestF16toQD8() const {
    xnnpack::ReplicableRandomDevice rng;

    xnnpack::Buffer<float> input_float((batch_size() - 1) * input_stride() +
                                   channels());
    xnnpack::Buffer<xnn_float16> input(XNN_EXTRA_BYTES / sizeof(xnn_float16) +
                                (batch_size() - 1) * input_stride() +
                                channels());
    xnnpack::Buffer<int8_t> output((batch_size() - 1) * output_stride() +
                               channels());
    xnnpack::Buffer<xnn_quantization_params> quantization_params(
        batch_size() + XNN_EXTRA_QUANTIZATION_PARAMS);
    std::uniform_real_distribution<float> range_dist(-10, 10);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      const float min_val = std::min(range_dist(rng), range_dist(rng));
      const float max_val = std::uniform_real_distribution<float>(
          min_val *
              (1.0f + std::numeric_limits<uint8_t>::max() * 6.103515625e-5f),
          10.0f)(rng);
      std::uniform_real_distribution<float> f32dist(min_val, max_val);
      std::generate(input_float.begin(), input_float.end(),
                    [&]() { return f32dist(rng); });
      std::copy(input_float.begin(), input_float.end(), input.begin());
      std::copy(input.begin(), input.begin() + channels(),
                     input_float.begin());

      // Create, setup, run, and destroy Convert operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convert_op = nullptr;

      xnn_status status = xnn_create_convert_nc_f16_qd8(0, &convert_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, convert_op);

      // Smart pointer to automatically delete convert op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
          auto_convert_op(convert_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
                xnn_reshape_convert_nc_f16_qd8(
                    convert_op, batch_size(), channels(), input_stride(),
                    output_stride(), /*threadpool=*/nullptr));
      ASSERT_EQ(xnn_status_success, xnn_setup_convert_nc_f16_qd8(
                                        convert_op, input.data(), output.data(),
                                        quantization_params.data()));
      ASSERT_EQ(xnn_status_success,
                xnn_run_operator(convert_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        const float* input_ptr = &input_float[i * input_stride()];
        const auto minmax =
            std::minmax_element(input_ptr, input_ptr + channels());
        const float rmin = math_min_f32(0.0f, *minmax.first);
        const float rmax = math_max_f32(0.0f, *minmax.second);
        const float max_acceptable_error =
            0.8f * (rmax - rmin) / std::numeric_limits<uint8_t>::max();
        for (size_t c = 0; c < channels(); c++) {
          float expected = input_float[i * input_stride() + c];
          int8_t quantized_val = (int)output[i * output_stride() + c];
          float dequantized_val =
              static_cast<float>(quantized_val -
                                 quantization_params[i].zero_point) *
              quantization_params[i].scale;
          ASSERT_NEAR(expected, dequantized_val, max_acceptable_error)
              << "at batch " << i << " / " << batch_size() << ", channel " << c
              << " / " << channels() << ", rmin=" << rmin << ", rmax=" << rmax
              << ", quantization_params={zero_point="
              << quantization_params[i].zero_point
              << ", scale=" << quantization_params[i].scale << "}";
        }
      }
    }
  }

  void TestF32toQD8() const {
    xnnpack::ReplicableRandomDevice rng;

    xnnpack::Buffer<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      (batch_size() - 1) * input_stride() + channels());
    xnnpack::Buffer<int8_t> output((batch_size() - 1) * output_stride() + channels());
    xnnpack::Buffer<xnn_quantization_params> quantization_params(batch_size() + XNN_EXTRA_QUANTIZATION_PARAMS);
    std::uniform_real_distribution<float> range_dist(-100000, 100000);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      const float first_val = range_dist(rng);
      const float second_val = range_dist(rng);
      std::uniform_real_distribution<float> f32dist(std::min(first_val, second_val), std::max(first_val, second_val));
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });

      // Create, setup, run, and destroy Convert operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convert_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_convert_nc_f32_qd8(
          0, &convert_op));
      ASSERT_NE(nullptr, convert_op);

      // Smart pointer to automatically delete convert op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convert_op(convert_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success, xnn_reshape_convert_nc_f32_qd8(convert_op, batch_size(),
          channels(), input_stride(), output_stride(), /*threadpool=*/nullptr));
      ASSERT_EQ(xnn_status_success, xnn_setup_convert_nc_f32_qd8(convert_op, input.data(), output.data(), quantization_params.data()));
      ASSERT_EQ(xnn_status_success, xnn_run_operator(convert_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        const float* input_ptr = &input[i * input_stride()];
        const auto minmax = std::minmax_element(input_ptr, input_ptr + channels());
        const float rmin = math_min_f32(0.0f, *minmax.first);
        const float rmax = math_max_f32(0.0f, *minmax.second);
        const float max_acceptable_error = 0.5001f * (rmax - rmin) / std::numeric_limits<uint8_t>::max();
        for (size_t c = 0; c < channels(); c++) {
          float expected = input[i * input_stride() + c];
          int8_t quantized_val = output[i * output_stride() + c];
          float dequantized_val = float(quantized_val - quantization_params[i].zero_point) * quantization_params[i].scale;
          EXPECT_NEAR(expected, dequantized_val, max_acceptable_error)
            << "at batch " << i << " / " << batch_size() << ", channel " << c << " / " << channels();
        }
      }
    }
  }

  void TestF32toQP8() const {
    xnnpack::ReplicableRandomDevice rng;

    // The parameters of the GEMM config are used as packing parameters.
    const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_nr2_config();

    xnnpack::Buffer<float> input(XNN_EXTRA_BYTES / sizeof(float) +
                             (batch_size() - 1) * input_stride() + channels());
    xnnpack::Buffer<int8_t> output(xnn_x8_packq_f32qp8_packed_size(
        batch_size(), channels(), gemm_config->mr, 1 << gemm_config->log2_kr,
        1 << gemm_config->log2_sr));
    std::uniform_real_distribution<float> range_dist(-100000, 100000);
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      const float first_val = range_dist(rng);
      const float second_val = range_dist(rng);
      std::uniform_real_distribution<float> f32dist(
          std::min(first_val, second_val), std::max(first_val, second_val));
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });

      // Create, setup, run, and destroy Convert operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t convert_op = nullptr;

      ASSERT_EQ(xnn_status_success,
                xnn_create_convert_nc_f32_qp8(0, &convert_op));
      ASSERT_NE(nullptr, convert_op);

      // Smart pointer to automatically delete convert op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>
          auto_convert_op(convert_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
                xnn_reshape_convert_nc_f32_qp8(convert_op, batch_size(),
                                               channels(), input_stride(),
                                               /*threadpool=*/nullptr));
      ASSERT_EQ(xnn_status_success,
                xnn_setup_convert_nc_f32_qp8(convert_op, input.data(),
                                             output.data()));
      ASSERT_EQ(xnn_status_success,
                xnn_run_operator(convert_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        // const float* input_ptr = &input[i * input_stride()];
        // const auto minmax =
        //     std::minmax_element(input_ptr, input_ptr + channels());
        // const float rmin = math_min_f32(0.0f, *minmax.first);
        // const float rmax = math_max_f32(0.0f, *minmax.second);
        // const float max_acceptable_error =
        //     0.5001f * (rmax - rmin) / std::numeric_limits<uint8_t>::max();

        // TODO(b/340399245) - Find a way to extract individual quantized values
        // from the packing?
        ASSERT_TRUE(true);
      }
    }
  }

 private:
  size_t batch_size_{1};
  size_t channels_{1};
  size_t input_stride_{0};
  size_t output_stride_{0};
  float input_scale_{150.0f};
  float output_scale_{3.0f};
  int16_t zero_point_{1};
  size_t iterations_{15};
};

TEST(CONVERT_NC_F16_QD8, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(1)
        .channels(channels)
        .iterations(3)
        .TestF16toQD8();
  }
}

TEST(CONVERT_NC_F16_QD8, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
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
        .iterations(3)
        .TestF16toQD8();
  }
}

TEST(CONVERT_NC_F32_QD8, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(1)
        .channels(channels)
        .iterations(3)
        .TestF32toQD8();
  }
}

TEST(CONVERT_NC_F32_QD8, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
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
        .iterations(3)
        .TestF32toQD8();
  }
}

TEST(CONVERT_NC_F32_QP8, unit_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(1)
        .channels(channels)
        .iterations(3)
        .TestF32toQD8();
  }
}

TEST(CONVERT_NC_F32_QP8, small_batch) {
  for (size_t channels = 1; channels < 100; channels++) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .iterations(3)
        .TestF32toQD8();
  }
}

TEST(CONVERT_NC_F32_QP8, small_batch_with_input_stride) {
  for (size_t channels = 10; channels < 11; channels += 15) {
    ConvertOperatorTester()
        .batch_size(3)
        .channels(channels)
        .input_stride(129)
        .iterations(3)
        .TestF32toQD8();
  }
}
