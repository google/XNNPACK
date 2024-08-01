// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "vbinary-microkernel-tester.h"

#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <vector>
#include <climits>

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/requantization.h"
#include "replicable_random_device.h"

void VBinaryMicrokernelTester::Test(xnn_f16_vbinary_ukernel_fn vbinary,
                                    OpType op_type) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(0.01f, 1.0f);

  std::vector<uint16_t> a(batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::vector<uint16_t> b(batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::vector<uint16_t> y(
      batch_size() +
      (inplace_a() || inplace_b() ? XNN_EXTRA_BYTES / sizeof(uint16_t) : 0));
  std::vector<float> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(),
                  [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
    std::generate(b.begin(), b.end(),
                  [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
    if (inplace_a() || inplace_b()) {
      std::generate(y.begin(), y.end(),
                    [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
    } else {
      std::fill(y.begin(), y.end(), UINT16_C(0x7E00) /* NaN */);
    }
    const uint16_t* a_data = inplace_a() ? y.data() : a.data();
    const uint16_t* b_data = inplace_b() ? y.data() : b.data();

    // Compute reference results.
    for (size_t i = 0; i < batch_size(); i++) {
      switch (op_type) {
        case OpType::Add:
          y_ref[i] = fp16_ieee_to_fp32_value(a_data[i]) +
                     fp16_ieee_to_fp32_value(b_data[i]);
          break;
        case OpType::CopySign:
          y_ref[i] = std::copysign(fp16_ieee_to_fp32_value(a_data[i]), fp16_ieee_to_fp32_value(b_data[i]));
          break;
        case OpType::Div:
          y_ref[i] = fp16_ieee_to_fp32_value(a_data[i]) /
                     fp16_ieee_to_fp32_value(b_data[i]);
          break;
        case OpType::Max:
          y_ref[i] = std::max<float>(fp16_ieee_to_fp32_value(a_data[i]),
                                     fp16_ieee_to_fp32_value(b_data[i]));
          break;
        case OpType::Min:
          y_ref[i] = std::min<float>(fp16_ieee_to_fp32_value(a_data[i]),
                                     fp16_ieee_to_fp32_value(b_data[i]));
          break;
        case OpType::Mul:
          y_ref[i] = fp16_ieee_to_fp32_value(a_data[i]) *
                     fp16_ieee_to_fp32_value(b_data[i]);
          break;
        case OpType::SqrDiff: {
          const float diff = fp16_ieee_to_fp32_value(a_data[i]) -
                             fp16_ieee_to_fp32_value(b_data[i]);
          y_ref[i] = diff * diff;
          break;
        }
        case OpType::Sub:
          y_ref[i] = fp16_ieee_to_fp32_value(a_data[i]) -
                     fp16_ieee_to_fp32_value(b_data[i]);
          break;
      }
    }

    // Call optimized micro-kernel.
    vbinary(batch_size() * sizeof(uint16_t), a_data, b_data, y.data(), nullptr);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      EXPECT_NEAR(fp16_ieee_to_fp32_value(y[i]), y_ref[i],
                  std::max(1.0e-4f, std::abs(y_ref[i]) * 1.0e-2f))
          << "at " << i << " / " << batch_size();
    }
  }
}

void VBinaryMicrokernelTester::Test(
    xnn_f16_vbinary_minmax_ukernel_fn vbinary_minmax, OpType op_type,
    xnn_init_f16_minmax_params_fn init_params) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(0.01f, 1.0f);

  std::vector<uint16_t> a(batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::vector<uint16_t> b(batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::vector<uint16_t> y(
      batch_size() +
      (inplace_a() || inplace_b() ? XNN_EXTRA_BYTES / sizeof(uint16_t) : 0));
  std::vector<float> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(),
                  [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
    std::generate(b.begin(), b.end(),
                  [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
    if (inplace_a() || inplace_b()) {
      std::generate(y.begin(), y.end(),
                    [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
    } else {
      std::fill(y.begin(), y.end(), UINT16_C(0x7E00) /* NaN */);
    }
    const uint16_t* a_data = inplace_a() ? y.data() : a.data();
    const uint16_t* b_data = inplace_b() ? y.data() : b.data();

    // Compute reference results.
    for (size_t i = 0; i < batch_size(); i++) {
      switch (op_type) {
        case OpType::Add:
          y_ref[i] = fp16_ieee_to_fp32_value(a_data[i]) +
                     fp16_ieee_to_fp32_value(b_data[i]);
          break;
        case OpType::CopySign:
          y_ref[i] = std::copysign(fp16_ieee_to_fp32_value(a_data[i]), fp16_ieee_to_fp32_value(b_data[i]));
          break;
        case OpType::Div:
          y_ref[i] = fp16_ieee_to_fp32_value(a_data[i]) /
                     fp16_ieee_to_fp32_value(b_data[i]);
          break;
        case OpType::Max:
          y_ref[i] = std::max<float>(fp16_ieee_to_fp32_value(a_data[i]),
                                     fp16_ieee_to_fp32_value(b_data[i]));
          break;
        case OpType::Min:
          y_ref[i] = std::min<float>(fp16_ieee_to_fp32_value(a_data[i]),
                                     fp16_ieee_to_fp32_value(b_data[i]));
          break;
        case OpType::Mul:
          y_ref[i] = fp16_ieee_to_fp32_value(a_data[i]) *
                     fp16_ieee_to_fp32_value(b_data[i]);
          break;
        case OpType::SqrDiff: {
          const float diff = fp16_ieee_to_fp32_value(a_data[i]) -
                             fp16_ieee_to_fp32_value(b_data[i]);
          y_ref[i] = diff * diff;
          break;
        }
        case OpType::Sub:
          y_ref[i] = fp16_ieee_to_fp32_value(a_data[i]) -
                     fp16_ieee_to_fp32_value(b_data[i]);
          break;
      }
    }

    const float accumulated_min =
        *std::min_element(y_ref.cbegin(), y_ref.cend());
    const float accumulated_max =
        *std::max_element(y_ref.cbegin(), y_ref.cend());
    const float accumulated_range = accumulated_max - accumulated_min;
    const float y_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(
        accumulated_range > 0.0f
            ? (accumulated_max -
               accumulated_range / 255.0f * static_cast<float>(255 - qmax()))
            : +std::numeric_limits<float>::infinity()));
    const float y_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(
        accumulated_range > 0.0f
            ? (accumulated_min +
               accumulated_range / 255.0f * static_cast<float>(qmin()))
            : -std::numeric_limits<float>::infinity()));
    for (size_t i = 0; i < batch_size(); i++) {
      y_ref[i] = std::max<float>(std::min<float>(y_ref[i], y_max), y_min);
    }

    // Prepare parameters.
    xnn_f16_minmax_params params;
    init_params(&params, fp16_ieee_from_fp32_value(y_min),
                fp16_ieee_from_fp32_value(y_max));

    // Call optimized micro-kernel.
    vbinary_minmax(batch_size() * sizeof(uint16_t), a_data, b_data, y.data(),
                   &params);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      EXPECT_NEAR(fp16_ieee_to_fp32_value(y[i]), y_ref[i],
                  std::max(1.0e-4f, std::abs(y_ref[i]) * 1.0e-2f))
          << "at " << i << " / " << batch_size();
    }
  }
}

void VBinaryMicrokernelTester::Test(
    xnn_f32_vbinary_ukernel_fn vbinary, OpType op_type,
    xnn_init_f32_default_params_fn init_params) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

  std::vector<float> a(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> y(batch_size() + (inplace_a() || inplace_b()
                                           ? XNN_EXTRA_BYTES / sizeof(float)
                                           : 0));
  std::vector<float> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    if (inplace_a() || inplace_b()) {
      std::generate(y.begin(), y.end(), [&]() { return f32dist(rng); });
    } else {
      std::fill(y.begin(), y.end(), nanf(""));
    }
    const float* a_data = inplace_a() ? y.data() : a.data();
    const float* b_data = inplace_b() ? y.data() : b.data();

    // Compute reference results.
    for (size_t i = 0; i < batch_size(); i++) {
      switch (op_type) {
        case OpType::Add:
          y_ref[i] = a_data[i] + b_data[i];
          break;
        case OpType::CopySign:
          y_ref[i] = std::copysign(a_data[i], b_data[i]);
          break;
        case OpType::Div:
          y_ref[i] = a_data[i] / b_data[i];
          break;
        case OpType::Max:
          y_ref[i] = std::max<float>(a_data[i], b_data[i]);
          break;
        case OpType::Min:
          y_ref[i] = std::min<float>(a_data[i], b_data[i]);
          break;
        case OpType::Mul:
          y_ref[i] = a_data[i] * b_data[i];
          break;
        case OpType::SqrDiff: {
          const float diff = a_data[i] - b_data[i];
          y_ref[i] = diff * diff;
          break;
        }
        case OpType::Sub:
          y_ref[i] = a_data[i] - b_data[i];
          break;
      }
    }

    // Prepare parameters.
    xnn_f32_default_params params;
    if (init_params != nullptr) {
      init_params(&params);
    }

    // Call optimized micro-kernel.
    vbinary(batch_size() * sizeof(float), a_data, b_data, y.data(),
            init_params != nullptr ? &params : nullptr);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      EXPECT_NEAR(y[i], y_ref[i], std::abs(y_ref[i]) * 1.0e-6f)
          << "at " << i << " / " << batch_size();
    }
  }
}

void VBinaryMicrokernelTester::Test(
    xnn_s32_vbinary_ukernel_fn vbinary, OpType op_type,
    xnn_init_s32_default_params_fn init_params) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_int_distribution<int32_t> s32dist(-100000, 100000);

  std::vector<int32_t> a(batch_size() + XNN_EXTRA_BYTES / sizeof(int32_t));
  std::vector<int32_t> b(batch_size() + XNN_EXTRA_BYTES / sizeof(int32_t));
  std::vector<int32_t> y(batch_size() + (inplace_a() || inplace_b()
                                           ? XNN_EXTRA_BYTES / sizeof(int32_t)
                                           : 0));
  std::vector<int32_t> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return s32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return s32dist(rng); });
    if (inplace_a() || inplace_b()) {
      std::generate(y.begin(), y.end(), [&]() { return s32dist(rng); });
    } else {
      std::fill(y.begin(), y.end(), INT_MAX);
    }
    const int32_t* a_data = inplace_a() ? y.data() : a.data();
    const int32_t* b_data = inplace_b() ? y.data() : b.data();

    // Compute reference results.
    for (size_t i = 0; i < batch_size(); i++) {
        switch (op_type) {
        case OpType::Add:
          y_ref[i] = a_data[i] + b_data[i];
          break;
        case OpType::CopySign:
          y_ref[i] = std::copysign(a_data[i], b_data[i]);
          break;
        case OpType::Div:
          y_ref[i] = a_data[i] / b_data[i];
          break;
        case OpType::Max:
          y_ref[i] = std::max<int32_t>(a_data[i], b_data[i]);
          break;
        case OpType::Min:
          y_ref[i] = std::min<int32_t>(a_data[i], b_data[i]);
          break;
        case OpType::Mul:
          // Overflow is the expected behaviour
          y_ref[i] = ((((int64_t) a_data[i] * (int64_t) b_data[i]) << 32) >> 32);
          break;
        case OpType::SqrDiff: {
          const int32_t diff = a_data[i] - b_data[i];
          y_ref[i] = diff * diff;
          break;
        }
        case OpType::Sub:
          y_ref[i] = a_data[i] - b_data[i];
          break;
      }
    }


    // Prepare parameters.
    xnn_s32_default_params params;
    if (init_params != nullptr) {
      init_params(&params);
    }

    // Call optimized micro-kernel.
    vbinary(batch_size() * sizeof(int32_t), a_data, b_data, y.data(),
            &params);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      EXPECT_EQ(y[i], y_ref[i])
          << "at " << i << " / " << batch_size();
    }
  }
}

void VBinaryMicrokernelTester::Test(
    xnn_f32_vbinary_relu_ukernel_fn vbinary_relu, OpType op_type) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> lhs_f32dist(-1.0f, 1.0f);
  // For denominator, avoid 0 so we don't get Infinity as the result.
  std::uniform_real_distribution<float> rhs_f32dist(0.1, 1.0f);

  std::vector<float> a(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> y(batch_size() + (inplace_a() || inplace_b()
                                           ? XNN_EXTRA_BYTES / sizeof(float)
                                           : 0));
  std::vector<float> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return lhs_f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return rhs_f32dist(rng); });
    if (inplace_a() || inplace_b()) {
      std::generate(y.begin(), y.end(), [&]() { return lhs_f32dist(rng); });
    } else {
      std::fill(y.begin(), y.end(), nanf(""));
    }
    const float* a_data = inplace_a() ? y.data() : a.data();
    const float* b_data = inplace_b() ? y.data() : b.data();

    // Compute reference results.
    for (size_t i = 0; i < batch_size(); i++) {
      switch (op_type) {
        case OpType::Add:
          y_ref[i] = a_data[i] + b_data[i];
          break;
        case OpType::CopySign:
          y_ref[i] = std::copysign(a_data[i], b_data[i]);
          break;
        case OpType::Div:
          y_ref[i] = a_data[i] / b_data[i];
          break;
        case OpType::Max:
          y_ref[i] = std::max<float>(a_data[i], b_data[i]);
          break;
        case OpType::Min:
          y_ref[i] = std::min<float>(a_data[i], b_data[i]);
          break;
        case OpType::Mul:
          y_ref[i] = a_data[i] * b_data[i];
          break;
        case OpType::SqrDiff: {
          const float diff = a_data[i] - b_data[i];
          y_ref[i] = diff * diff;
          break;
        }
        case OpType::Sub:
          y_ref[i] = a_data[i] - b_data[i];
          break;
      }
    }
    for (size_t i = 0; i < batch_size(); i++) {
      y_ref[i] = std::max(y_ref[i], 0.0f);
    }

    // Call optimized micro-kernel.
    vbinary_relu(batch_size() * sizeof(float), a_data, b_data, y.data(),
                 nullptr);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      EXPECT_GE(y[i], 0.0f) << "at " << i << " / " << batch_size();
      EXPECT_NEAR(y[i], y_ref[i], std::abs(y_ref[i]) * 1.0e-6f)
          << "at " << i << " / " << batch_size();
    }
  }
}

void VBinaryMicrokernelTester::Test(
    xnn_f32_vbinary_minmax_ukernel_fn vbinary_minmax, OpType op_type,
    xnn_init_f32_minmax_params_fn init_params) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(0.01f, 1.0f);

  std::vector<float> a(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> b(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> y(batch_size() + (inplace_a() || inplace_b()
                                           ? XNN_EXTRA_BYTES / sizeof(float)
                                           : 0));
  std::vector<float> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    if (inplace_a() || inplace_b()) {
      std::generate(y.begin(), y.end(), [&]() { return f32dist(rng); });
    } else {
      std::fill(y.begin(), y.end(), nanf(""));
    }
    const float* a_data = inplace_a() ? y.data() : a.data();
    const float* b_data = inplace_b() ? y.data() : b.data();

    // Compute reference results.
    for (size_t i = 0; i < batch_size(); i++) {
      switch (op_type) {
        case OpType::Add:
          y_ref[i] = a_data[i] + b_data[i];
          break;
        case OpType::CopySign:
          y_ref[i] = std::copysign(a_data[i], b_data[i]);
          break;
        case OpType::Div:
          y_ref[i] = a_data[i] / b_data[i];
          break;
        case OpType::Max:
          y_ref[i] = std::max<float>(a_data[i], b_data[i]);
          break;
        case OpType::Min:
          y_ref[i] = std::min<float>(a_data[i], b_data[i]);
          break;
        case OpType::Mul:
          y_ref[i] = a_data[i] * b_data[i];
          break;
        case OpType::SqrDiff: {
          const float diff = a_data[i] - b_data[i];
          y_ref[i] = diff * diff;
          break;
        }
        case OpType::Sub:
          y_ref[i] = a_data[i] - b_data[i];
          break;
      }
    }
    const float accumulated_min =
        *std::min_element(y_ref.cbegin(), y_ref.cend());
    const float accumulated_max =
        *std::max_element(y_ref.cbegin(), y_ref.cend());
    const float accumulated_range = accumulated_max - accumulated_min;
    const float y_max =
        accumulated_range > 0.0f
            ? (accumulated_max -
               accumulated_range / 255.0f * static_cast<float>(255 - qmax()))
            : +std::numeric_limits<float>::infinity();
    const float y_min = accumulated_range > 0.0f
                            ? (accumulated_min + accumulated_range / 255.0f *
                                                     static_cast<float>(qmin()))
                            : -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < batch_size(); i++) {
      y_ref[i] = std::max<float>(std::min<float>(y_ref[i], y_max), y_min);
    }

    // Prepare parameters.
    xnn_f32_minmax_params params;
    init_params(&params, y_min, y_max);

    // Call optimized micro-kernel.
    vbinary_minmax(batch_size() * sizeof(float), a_data, b_data, y.data(),
                   &params);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      EXPECT_NEAR(y[i], y_ref[i], std::abs(y_ref[i]) * 1.0e-6f)
          << "at " << i << " / " << batch_size();
    }
  }
}

void VBinaryMicrokernelTester::Test(
    xnn_qu8_vadd_minmax_ukernel_fn vadd_minmax,
    xnn_init_qu8_add_minmax_params_fn init_params) const {
  xnnpack::ReplicableRandomDevice rng;
  auto u8rng = [&rng]() {
    return std::uniform_int_distribution<uint32_t>(
        0, std::numeric_limits<uint8_t>::max())(rng);
  };

  std::vector<uint8_t> a(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<uint8_t> b(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<uint8_t> y(batch_size() + (inplace_a() || inplace_b()
                                             ? XNN_EXTRA_BYTES / sizeof(uint8_t)
                                             : 0));
  std::vector<float> y_fp(batch_size());
  std::vector<uint8_t> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(u8rng));
    std::generate(b.begin(), b.end(), std::ref(u8rng));
    if (inplace_a() || inplace_b()) {
      std::generate(y.begin(), y.end(), std::ref(u8rng));
    } else {
      std::fill(y.begin(), y.end(), 0xA5);
    }
    const uint8_t* a_data = inplace_a() ? y.data() : a.data();
    const uint8_t* b_data = inplace_b() ? y.data() : b.data();

    // Prepare parameters.
    xnn_qu8_add_minmax_params quantization_params;
    init_params(&quantization_params, a_zero_point(), b_zero_point(),
                y_zero_point(), a_scale() / y_scale(), b_scale() / y_scale(),
                qmin(), qmax());
    xnn_qu8_add_minmax_params scalar_quantization_params;
    xnn_init_qu8_add_minmax_scalar_params(
        &scalar_quantization_params, a_zero_point(), b_zero_point(),
        y_zero_point(), a_scale() / y_scale(), b_scale() / y_scale(), qmin(),
        qmax());

    // Compute reference results.
    for (size_t i = 0; i < batch_size(); i++) {
      y_fp[i] = static_cast<float>(y_zero_point()) +
                static_cast<float>(static_cast<int32_t>(a_data[i]) -
                                   static_cast<int32_t>(a_zero_point())) *
                    (a_scale() / y_scale()) +
                static_cast<float>(static_cast<int32_t>(b_data[i]) -
                                   static_cast<int32_t>(b_zero_point())) *
                    (b_scale() / y_scale());
      y_fp[i] = std::min<float>(y_fp[i], static_cast<float>(qmax()));
      y_fp[i] = std::max<float>(y_fp[i], static_cast<float>(qmin()));
      y_ref[i] = xnn_qu8_quantize_add(a_data[i], b_data[i],
                                      scalar_quantization_params);
    }

    // Call optimized micro-kernel.
    vadd_minmax(batch_size(), a_data, b_data, y.data(), &quantization_params);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      EXPECT_LE(static_cast<uint32_t>(y[i]), static_cast<uint32_t>(qmax()))
          << "at element " << i << " / " << batch_size();
      EXPECT_GE(static_cast<uint32_t>(y[i]), static_cast<uint32_t>(qmin()))
          << "at element " << i << " / " << batch_size();
      EXPECT_NEAR(static_cast<float>(static_cast<int32_t>(y[i])), y_fp[i], 0.6f)
          << "at element " << i << " / " << batch_size();
      EXPECT_EQ(static_cast<uint32_t>(y_ref[i]), static_cast<uint32_t>(y[i]))
          << "at element " << i << " / " << batch_size();
    }
  }
}

void VBinaryMicrokernelTester::Test(
    xnn_qu8_vmul_minmax_ukernel_fn vmul_minmax,
    xnn_init_qu8_mul_minmax_params_fn init_params,
    xnn_qu8_requantize_fn requantize) const {
  xnnpack::ReplicableRandomDevice rng;
  auto u8rng = [&rng]() {
    return std::uniform_int_distribution<uint32_t>(
        0, std::numeric_limits<uint8_t>::max())(rng);
  };

  std::vector<uint8_t> a(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<uint8_t> b(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<uint8_t> y(batch_size() + (inplace_a() || inplace_b()
                                             ? XNN_EXTRA_BYTES / sizeof(uint8_t)
                                             : 0));
  std::vector<float> y_fp(batch_size());
  std::vector<uint8_t> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(u8rng));
    std::generate(b.begin(), b.end(), std::ref(u8rng));
    if (inplace_a() || inplace_b()) {
      std::generate(y.begin(), y.end(), std::ref(u8rng));
    } else {
      std::fill(y.begin(), y.end(), 0xA5);
    }
    const uint8_t* a_data = inplace_a() ? y.data() : a.data();
    const uint8_t* b_data = inplace_b() ? y.data() : b.data();

    // Prepare parameters.
    const float product_scale = a_scale() * b_scale();
    const float product_output_scale = product_scale / y_scale();
    xnn_qu8_mul_minmax_params quantization_params;
    init_params(&quantization_params, a_zero_point(), b_zero_point(),
                y_zero_point(), product_output_scale, qmin(), qmax());

    // Compute reference results.
    for (size_t i = 0; i < batch_size(); i++) {
      const int32_t acc = (static_cast<int32_t>(a_data[i]) -
                           static_cast<int32_t>(a_zero_point())) *
                          (static_cast<int32_t>(b_data[i]) -
                           static_cast<int32_t>(b_zero_point()));
      y_fp[i] = static_cast<float>(y_zero_point()) +
                product_output_scale * static_cast<float>(acc);
      y_fp[i] = std::min<float>(
          y_fp[i], static_cast<float>(static_cast<int32_t>(qmax())));
      y_fp[i] = std::max<float>(
          y_fp[i], static_cast<float>(static_cast<int32_t>(qmin())));
      y_ref[i] =
          requantize(acc, product_output_scale, y_zero_point(), qmin(), qmax());
    }

    // Call optimized micro-kernel.
    vmul_minmax(batch_size(), a_data, b_data, y.data(), &quantization_params);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      EXPECT_LE(static_cast<uint32_t>(y[i]), static_cast<uint32_t>(qmax()))
          << "at element " << i << " / " << batch_size();
      EXPECT_GE(static_cast<uint32_t>(y[i]), static_cast<uint32_t>(qmin()))
          << "at element " << i << " / " << batch_size();
      EXPECT_NEAR(static_cast<float>(static_cast<int32_t>(y[i])), y_fp[i], 0.6f)
          << "at element " << i << " / " << batch_size();
      EXPECT_EQ(static_cast<uint32_t>(y[i]), static_cast<uint32_t>(y_ref[i]))
          << "at element " << i << " / " << batch_size();
    }
  }
}

void VBinaryMicrokernelTester::Test(
    xnn_qs8_vadd_minmax_ukernel_fn vadd_minmax,
    xnn_init_qs8_add_minmax_params_fn init_params) const {
  xnnpack::ReplicableRandomDevice rng;
  auto i8rng = [&rng]() {
    return std::uniform_int_distribution<int32_t>(
        std::numeric_limits<int8_t>::min(),
        std::numeric_limits<int8_t>::max())(rng);
  };

  std::vector<int8_t> a(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<int8_t> b(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<int8_t> y(batch_size() + (inplace_a() || inplace_b()
                                            ? XNN_EXTRA_BYTES / sizeof(int8_t)
                                            : 0));
  std::vector<float> y_fp(batch_size());
  std::vector<int8_t> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(i8rng));
    std::generate(b.begin(), b.end(), std::ref(i8rng));
    if (inplace_a() || inplace_b()) {
      std::generate(y.begin(), y.end(), std::ref(i8rng));
    } else {
      std::fill(y.begin(), y.end(), 0xA5);
    }
    const int8_t* a_data = inplace_a() ? y.data() : a.data();
    const int8_t* b_data = inplace_b() ? y.data() : b.data();

    // Prepare parameters.
    xnn_qs8_add_minmax_params quantization_params;
    init_params(
        &quantization_params, static_cast<int8_t>(a_zero_point() - 0x80),
        static_cast<int8_t>(b_zero_point() - 0x80),
        static_cast<int8_t>(y_zero_point() - 0x80), a_scale() / y_scale(),
        b_scale() / y_scale(), static_cast<int8_t>(qmin() - 0x80),
        static_cast<int8_t>(qmax() - 0x80));
    xnn_qs8_add_minmax_params scalar_quantization_params;
    xnn_init_qs8_add_minmax_scalar_params(
        &scalar_quantization_params, static_cast<int8_t>(a_zero_point() - 0x80),
        static_cast<int8_t>(b_zero_point() - 0x80),
        static_cast<int8_t>(y_zero_point() - 0x80), a_scale() / y_scale(),
        b_scale() / y_scale(), static_cast<int8_t>(qmin() - 0x80),
        static_cast<int8_t>(qmax() - 0x80));

    // Compute reference results.
    for (size_t i = 0; i < batch_size(); i++) {
      y_fp[i] =
          static_cast<float>(static_cast<int32_t>(y_zero_point() - 0x80)) +
          static_cast<float>(static_cast<int32_t>(a_data[i]) -
                             static_cast<int32_t>(a_zero_point() - 0x80)) *
              (a_scale() / y_scale()) +
          static_cast<float>(static_cast<int32_t>(b_data[i]) -
                             static_cast<int32_t>(b_zero_point() - 0x80)) *
              (b_scale() / y_scale());
      y_fp[i] = std::min<float>(
          y_fp[i], static_cast<float>(static_cast<int32_t>(qmax() - 0x80)));
      y_fp[i] = std::max<float>(
          y_fp[i], static_cast<float>(static_cast<int32_t>(qmin() - 0x80)));
      y_ref[i] = xnn_qs8_quantize_add(a_data[i], b_data[i],
                                      scalar_quantization_params);
    }

    // Call optimized micro-kernel.
    vadd_minmax(batch_size(), a_data, b_data, y.data(), &quantization_params);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      EXPECT_LE(static_cast<int32_t>(y[i]), static_cast<int32_t>(qmax() - 0x80))
          << "at element " << i << " / " << batch_size();
      EXPECT_GE(static_cast<int32_t>(y[i]), static_cast<int32_t>(qmin() - 0x80))
          << "at element " << i << " / " << batch_size();
      EXPECT_EQ(static_cast<int32_t>(y_ref[i]), static_cast<int32_t>(y[i]))
          << "at element " << i << " / " << batch_size();
      EXPECT_NEAR(static_cast<float>(static_cast<int32_t>(y[i])), y_fp[i], 0.6f)
          << "at element " << i << " / " << batch_size();
    }
  }
}

void VBinaryMicrokernelTester::Test(
    xnn_qs8_vmul_minmax_ukernel_fn vmul_minmax,
    xnn_init_qs8_mul_minmax_params_fn init_params,
    xnn_qs8_requantize_fn requantize) const {
  xnnpack::ReplicableRandomDevice rng;
  auto i8rng = [&rng]() {
    return std::uniform_int_distribution<int32_t>(
        std::numeric_limits<int8_t>::min(),
        std::numeric_limits<int8_t>::max())(rng);
  };

  std::vector<int8_t> a(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<int8_t> b(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<int8_t> y(batch_size() + (inplace_a() || inplace_b()
                                            ? XNN_EXTRA_BYTES / sizeof(int8_t)
                                            : 0));
  std::vector<float> y_fp(batch_size());
  std::vector<int8_t> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(a.begin(), a.end(), std::ref(i8rng));
    std::generate(b.begin(), b.end(), std::ref(i8rng));
    if (inplace_a() || inplace_b()) {
      std::generate(y.begin(), y.end(), std::ref(i8rng));
    } else {
      std::fill(y.begin(), y.end(), 0xA5);
    }
    const int8_t* a_data = inplace_a() ? y.data() : a.data();
    const int8_t* b_data = inplace_b() ? y.data() : b.data();

    // Prepare parameters.
    const float product_scale = a_scale() * b_scale();
    const float product_output_scale = product_scale / y_scale();
    EXPECT_GE(product_output_scale, 0x1.0p-32f);
    xnn_qs8_mul_minmax_params quantization_params;
    init_params(
        &quantization_params, static_cast<int8_t>(a_zero_point() - 0x80),
        static_cast<int8_t>(b_zero_point() - 0x80),
        static_cast<int8_t>(y_zero_point() - 0x80), product_output_scale,
        static_cast<int8_t>(qmin() - 0x80), static_cast<int8_t>(qmax() - 0x80));

    // Compute reference results.
    for (size_t i = 0; i < batch_size(); i++) {
      const int32_t acc = (static_cast<int32_t>(a_data[i]) -
                           static_cast<int32_t>(a_zero_point() - 0x80)) *
                          (static_cast<int32_t>(b_data[i]) -
                           static_cast<int32_t>(b_zero_point() - 0x80));
      y_fp[i] = static_cast<float>(y_zero_point() - 0x80) +
                product_output_scale * static_cast<float>(acc);
      y_fp[i] = std::min<float>(
          y_fp[i], static_cast<float>(static_cast<int32_t>(qmax() - 0x80)));
      y_fp[i] = std::max<float>(
          y_fp[i], static_cast<float>(static_cast<int32_t>(qmin() - 0x80)));
      y_ref[i] = requantize(acc, product_output_scale,
                            static_cast<int8_t>(y_zero_point() - 0x80),
                            static_cast<int8_t>(qmin() - 0x80),
                            static_cast<int8_t>(qmax() - 0x80));
    }

    // Call optimized micro-kernel.
    vmul_minmax(batch_size(), a_data, b_data, y.data(), &quantization_params);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      EXPECT_LE(static_cast<int32_t>(y[i]), static_cast<int32_t>(qmax() - 0x80))
          << "at element " << i << " / " << batch_size();
      EXPECT_GE(static_cast<int32_t>(y[i]), static_cast<int32_t>(qmin() - 0x80))
          << "at element " << i << " / " << batch_size();
      EXPECT_EQ(static_cast<int32_t>(y_ref[i]), static_cast<int32_t>(y[i]))
          << "at element " << i << " / " << batch_size();
      EXPECT_NEAR(static_cast<float>(static_cast<int32_t>(y[i])), y_fp[i], 0.6f)
          << "at element " << i << " / " << batch_size();
    }
  }
}
