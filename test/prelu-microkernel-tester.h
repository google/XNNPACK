// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/microfnptr.h"
#include "replicable_random_device.h"
#include "xnnpack/microparams.h"
#include "xnnpack/pack.h"

class PReLUMicrokernelTester {
 public:
  PReLUMicrokernelTester& rows(size_t rows) {
    assert(rows != 0);
    this->rows_ = rows;
    return *this;
  }

  size_t rows() const {
    return this->rows_;
  }

  PReLUMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const {
    return this->channels_;
  }

  PReLUMicrokernelTester& input_stride(size_t input_stride) {
    assert(input_stride != 0);
    this->input_stride_ = input_stride;
    return *this;
  }

  size_t input_stride() const {
    if (this->input_stride_ == 0) {
      return channels();
    } else {
      assert(this->input_stride_ >= channels());
      return this->input_stride_;
    }
  }

  PReLUMicrokernelTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return channels();
    } else {
      assert(this->output_stride_ >= channels());
      return this->output_stride_;
    }
  }

  PReLUMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  bool inplace() const {
    return this->inplace_;
  }

  PReLUMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  float input_scale() const { return input_scale_; }

  int16_t input_zero_point() const { return input_zero_point_; }

  float slope_scale() const { return slope_scale_; }

  int16_t slope_zero_point() const { return slope_zero_point_; }

  float output_scale() const { return output_scale_; }

  int16_t output_zero_point() const { return output_zero_point_; }

  int16_t qmin() const { return qmin_; }

  int16_t qmax() const { return qmax_; }

  // Converters between float and quantized types.
  float FloatFromInputQS8(int8_t x) const {
    return input_scale() * (static_cast<int32_t>(x) -
                            static_cast<int32_t>(input_zero_point()));
  }

  float FloatFromInputQU8(uint8_t x) const {
    return input_scale() *
           (static_cast<int32_t>(x) - static_cast<int32_t>(input_zero_point()));
  }

  float FloatFromSlopeQS8(int8_t x) const {
    return slope_scale() *
           (static_cast<int32_t>(x) - static_cast<int32_t>(slope_zero_point()));
  }

  float FloatFromSlopeQU8(uint8_t x) const {
    return slope_scale() *
           (static_cast<int32_t>(x) - static_cast<int32_t>(slope_zero_point()));
  }

  float QuantizeAsFloatQS8(float x) const {
    float y =
        x / output_scale() + static_cast<int32_t>(output_zero_point());
    y = std::min<float>(y, qmax() - 0x80);
    y = std::max<float>(y, qmin() - 0x80);
    return y;
  }

  float QuantizeAsFloatQU8(float x) const {
    float y = x / output_scale() + static_cast<int32_t>(output_zero_point());
    y = std::min<float>(y, qmax());
    y = std::max<float>(y, qmin());
    return y;
  }

  void Test(xnn_f16_prelu_ukernel_fn prelu) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> w32dist(0.25f, 0.75f);

    std::vector<xnn_float16> x(channels() + (rows() - 1) * input_stride() + XNN_EXTRA_BYTES / sizeof(xnn_float16));
    std::vector<xnn_float16, AlignedAllocator<xnn_float16, 64>> w(channels() + XNN_EXTRA_BYTES / sizeof(xnn_float16));
    std::vector<xnn_float16> y(channels() + (rows() - 1) * output_stride() + XNN_EXTRA_BYTES / sizeof(xnn_float16));
    std::vector<float> y_ref(channels() * rows());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), [&]() { return f32dist(rng); });
      std::generate(w.begin(), w.end(), [&]() { return w32dist(rng); });
      if (inplace()) {
        std::generate(y.begin(), y.end(), [&]() { return f32dist(rng); });
      } else {
        std::fill(y.begin(), y.end(), std::nanf(""));
      }
      const xnn_float16* x_data = inplace() ? y.data() : x.data();

      // Compute reference results, without clamping.
      for (size_t n = 0; n < rows(); n++) {
        for (size_t c = 0; c < channels(); c++) {
          const float x_value = x_data[n * input_stride() + c];
          y_ref[n * channels() + c] = std::signbit(x_value) ?
              float(xnn_float16(x_value * w[c])) : x_value;  // What is going on here?
        }
      }

      // Call optimized micro-kernel.
      prelu(rows(), channels() * sizeof(xnn_float16),
        x_data, input_stride() * sizeof(xnn_float16),
        w.data(),
        y.data(), output_stride() * sizeof(xnn_float16));

      // Verify results.
      for (size_t n = 0; n < rows(); n++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_EQ(y[n * output_stride() + c], y_ref[n * channels() + c])
            << "at row " << n << " / " << rows()
            << ", channel " << c << " / " << channels();
        }
      }
    }
  }

  void Test(xnn_f32_prelu_ukernel_fn prelu) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> w32dist(0.25f, 0.75f);

    std::vector<float> x(channels() + (rows() - 1) * input_stride() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float, AlignedAllocator<float, 64>> w(channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(channels() + (rows() - 1) * output_stride() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y_ref(channels() * rows());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), [&]() { return f32dist(rng); });
      std::generate(w.begin(), w.end(), [&]() { return w32dist(rng); });
      if (inplace()) {
        std::generate(y.begin(), y.end(), [&]() { return f32dist(rng); });
      } else {
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* x_data = inplace() ? y.data() : x.data();

      // Compute reference results, without clamping.
      for (size_t n = 0; n < rows(); n++) {
        for (size_t c = 0; c < channels(); c++) {
          const float x_value = x_data[n * input_stride() + c];
          y_ref[n * channels() + c] = std::signbit(x_value) ? x_value * w[c] : x_value;
        }
      }

      // Call optimized micro-kernel.
      prelu(rows(), channels() * sizeof(float),
        x_data, input_stride() * sizeof(float),
        w.data(),
        y.data(), output_stride() * sizeof(float));

      // Verify results.
      for (size_t n = 0; n < rows(); n++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_EQ(y[n * output_stride() + c], y_ref[n * channels() + c])
            << "at row " << n << " / " << rows()
            << ", channel " << c << " / " << channels();
        }
      }
    }
  }

  void Test(xnn_qs8_prelu_ukernel_fn prelu, xnn_init_qs8_prelu_params_fn init_params) const {
      ASSERT_GE(input_zero_point(), std::numeric_limits<int8_t>::min());
      ASSERT_LE(input_zero_point(), std::numeric_limits<int8_t>::max());
      ASSERT_GE(output_zero_point(), std::numeric_limits<int8_t>::min());
      ASSERT_LE(output_zero_point(), std::numeric_limits<int8_t>::max());

      xnnpack::ReplicableRandomDevice rng;
      auto i8dist = std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(),
                    std::numeric_limits<int8_t>::max())(rng);
      
    std::vector<int8_t> x((rows() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t, AlignedAllocator<int8_t, 64>> w(channels() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> y((rows() - 1) * output_stride() + channels() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<float> y_ref(rows() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), [&] { return i8dist;});
      std::generate(w.begin(), w.end(), [&] { return i8dist;});
      std::fill(y.begin(), y.end(), 0xAA);

      // Compute reference results,
      for (size_t i = 0; i < rows(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          if((x[i * input_stride() + c] - (input_zero_point())) < 0) {
            y_ref[i * channels() + c] = QuantizeAsFloatQS8(FloatFromInputQS8(x[i * input_stride() + c]) * FloatFromSlopeQS8(w[c]));
          } else {
            y_ref[i * channels() + c] = QuantizeAsFloatQS8(FloatFromInputQS8(x[i * input_stride() + c]));
          }
        }
      }

      float positive_input_output_scale = input_scale() / output_scale();
      float negative_input_output_scale =  positive_input_output_scale * slope_scale();

      struct xnn_qs8_prelu_params params;
      init_params(&params, positive_input_output_scale, input_zero_point(), output_zero_point());

      int16_t weights_ptr[channels()];
      xnn_pack_qs8_prelu_w(channels(), channels(), w.data(), slope_zero_point(), &negative_input_output_scale, weights_ptr);


      prelu(rows(), channels()*sizeof(int8_t),
        x.data(), input_stride()*sizeof(int8_t),
        weights_ptr, y.data(), output_stride()*sizeof(int8_t), &params);
      
      for (size_t n = 0; n < rows(); n++) {
      for(size_t c = 0; c < channels(); c++) {
        ASSERT_NEAR(
              static_cast<float>(y[n * output_stride() + c]),
              y_ref[n * channels() + c], 1.5f)
            << "at position " << n << " / " << rows() << ", channel " << c << " / " << channels();
      }
      }
    }
  }

  void Test(xnn_qu8_prelu_ukernel_fn prelu, xnn_init_qu8_prelu_params_fn init_params) const {
      ASSERT_GE(input_zero_point(), std::numeric_limits<uint8_t>::min());
      ASSERT_LE(input_zero_point(), std::numeric_limits<uint8_t>::max());
      ASSERT_GE(output_zero_point(), std::numeric_limits<uint8_t>::min());
      ASSERT_LE(output_zero_point(), std::numeric_limits<uint8_t>::max());

      xnnpack::ReplicableRandomDevice rng;
      auto u8dist = std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(),
                    std::numeric_limits<uint8_t>::max())(rng);

      std::vector<uint8_t> x((rows() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint8_t));
      std::vector<uint8_t> w(channels());
      std::vector<uint8_t> y((rows() - 1) * output_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint8_t));
      std::vector<float> y_ref(rows() * channels());
      
      for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), [&] { return u8dist;});
      std::generate(w.begin(), w.end(), [&] { return u8dist;});
      std::fill(y.begin(), y.end(), 0xAA);

      // Compute reference results, without clamping.
      for (size_t i = 0; i < rows(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          if((x[i * input_stride() + c] - input_zero_point()) < 0) {
            y_ref[i * channels() + c] = QuantizeAsFloatQU8(FloatFromInputQU8(x[i * input_stride() + c]) * FloatFromSlopeQU8(w[c]));
          } else {
            y_ref[i * channels() + c] = QuantizeAsFloatQU8(FloatFromInputQU8(x[i * input_stride() + c]));
          }
        }
      }

      float positive_input_output_scale = input_scale() / output_scale();
      float negative_input_output_scale =  positive_input_output_scale * slope_scale();

      struct xnn_qu8_prelu_params params;
      init_params(&params, positive_input_output_scale, input_zero_point(), output_zero_point());

      int16_t weights_ptr[channels()];

      xnn_pack_qu8_prelu_w(channels(), channels(), w.data(), slope_zero_point(), &negative_input_output_scale, weights_ptr);

      prelu(rows(), channels()*sizeof(uint8_t),
        x.data(), input_stride()*sizeof(uint8_t),
        weights_ptr, y.data(), output_stride()*sizeof(uint8_t), &params);
      
      for (size_t n = 0; n < rows(); n++) {
      for(size_t c = 0; c < channels(); c++) {
        ASSERT_NEAR(
              static_cast<float>(y[n * output_stride() + c]),
              y_ref[n * channels() + c], 0.6f)
            << "at position " << n << " / " << rows() << ", channel " << c << " / " << channels();
      }
      }
      }
  }

 private:
  size_t rows_{1};
  size_t channels_{1};
  size_t input_stride_{0};
  size_t output_stride_{0};
  bool inplace_{false};
  float input_scale_{0.1f};
  int16_t input_zero_point_{1};
  float slope_scale_{0.05f};
  int16_t slope_zero_point_{1};
  float output_scale_{0.1f};
  int16_t output_zero_point_{5};
  int16_t qmin_{0};
  int16_t qmax_{255};
  size_t iterations_{15};
};
