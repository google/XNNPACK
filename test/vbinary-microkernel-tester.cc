// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "test/vbinary-microkernel-tester.h"

#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/requantization.h"
#include "test/replicable_random_device.h"

void VBinaryMicrokernelTester::Test(xnn_f16_vbinary_ukernel_fn vbinary,
                                    OpType op_type,
                                    xnn_init_f16_default_params_fn) const {
  xnnpack::ReplicableRandomDevice rng;
  xnnpack::DatatypeGenerator<xnn_float16> f16dist;

  const int stride_b = broadcast_b() ? 0 : 1;
  xnnpack::Buffer<xnn_float16> a(batch_size(), xnnpack::XnnExtraBytes);
  xnnpack::Buffer<xnn_float16> b(broadcast_b() ? 1 : batch_size(),
                                 xnnpack::XnnExtraBytes);
  xnnpack::Buffer<xnn_float16> y(
      batch_size(),
      xnnpack::PaddingBytes{inplace_a() || inplace_b() ? XNN_EXTRA_BYTES : 0});
  xnnpack::Buffer<xnn_float16> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    if (!inplace_a()) {
      std::generate(a.begin(), a.end(), [&]() { return f16dist(rng); });
    }
    if (!inplace_b()) {
      std::generate(b.begin(), b.end(), [&]() { return f16dist(rng); });
    }
    if (inplace_a() || inplace_b()) {
      std::generate(y.begin(), y.end(), [&]() { return f16dist(rng); });
    }
    const xnn_float16* a_data = inplace_a() ? y.data() : a.data();
    const xnn_float16* b_data = inplace_b() ? y.data() : b.data();
    reference_op_impl(a_data, b_data, y_ref.data(), batch_size(), op_type);

    // Call optimized micro-kernel.
    vbinary(batch_size() * sizeof(xnn_float16), a_data, b_data, y.data(),
            nullptr);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      if (std::isnan(y_ref[i])) {
        // TODO: We could check if y[i] is NaN, but not all our kernels do this.
      } else {
        ASSERT_NEAR(
            y[i], y_ref[i],
            std::max(1.0e-4f, std::abs(static_cast<float>(y_ref[i])) * 1.0e-2f))
            << "at " << i << " / " << batch_size()
            << ", a=" << static_cast<float>(a[i])
            << ", b=" << static_cast<float>(b[stride_b * i]);
      }
    }
  }
}

void VBinaryMicrokernelTester::Test(xnn_f32_vbinary_ukernel_fn vbinary,
                                    OpType op_type,
                                    xnn_init_f32_default_params_fn) const {
  xnnpack::ReplicableRandomDevice rng;
  xnnpack::DatatypeGenerator<float> f32dist;

  const int stride_b = broadcast_b() ? 0 : 1;
  xnnpack::Buffer<float> a(batch_size(), xnnpack::XnnExtraBytes);
  xnnpack::Buffer<float> b(broadcast_b() ? 1 : batch_size(),
                           xnnpack::XnnExtraBytes);
  xnnpack::Buffer<float> y(
      batch_size(),
      xnnpack::PaddingBytes{inplace_a() || inplace_b() ? XNN_EXTRA_BYTES : 0});
  xnnpack::Buffer<float> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    if (!inplace_a()) {
      std::generate(a.begin(), a.end(), [&]() { return f32dist(rng); });
    }
    if (!inplace_b()) {
      std::generate(b.begin(), b.end(), [&]() { return f32dist(rng); });
    }
    if (inplace_a() || inplace_b()) {
      std::generate(y.begin(), y.end(), [&]() { return f32dist(rng); });
    }
    const float* a_data = inplace_a() ? y.data() : a.data();
    const float* b_data = inplace_b() ? y.data() : b.data();
    reference_op_impl(a_data, b_data, y_ref.data(), batch_size(), op_type);

    // Call optimized micro-kernel.
    vbinary(batch_size() * sizeof(float), a_data, b_data, y.data(), nullptr);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      if (std::isnan(y_ref[i])) {
        // TODO: We could check if y[i] is NaN, but not all our kernels do this.
      } else {
        ASSERT_NEAR(y[i], y_ref[i], (std::abs(y_ref[i]) + 1.0f) * 1.0e-6f)
            << "at " << i << " / " << batch_size() << ", a=" << a[i]
            << ", b=" << b[stride_b * i];
      }
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

  xnnpack::Buffer<uint8_t> a(batch_size(), xnnpack::XnnExtraBytes);
  xnnpack::Buffer<uint8_t> b(broadcast_b() ? 1 : batch_size(),
                             xnnpack::XnnExtraBytes);
  xnnpack::Buffer<uint8_t> y(
      batch_size(),
      xnnpack::PaddingBytes{inplace_a() || inplace_b() ? XNN_EXTRA_BYTES : 0});
  xnnpack::Buffer<float> y_fp(batch_size());
  xnnpack::Buffer<uint8_t> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    if (!inplace_a()) {
      std::generate(a.begin(), a.end(), [&]() { return u8rng(); });
    }
    if (!inplace_b()) {
      std::generate(b.begin(), b.end(), [&]() { return u8rng(); });
    }
    if (inplace_a() || inplace_b()) {
      std::generate(y.begin(), y.end(), [&]() { return u8rng(); });
    }
    const uint8_t* a_data = inplace_a() ? y.data() : a.data();
    const uint8_t* b_data = inplace_b() ? y.data() : b.data();
    const size_t stride_b = broadcast_b() ? 0 : 1;

    // Prepare parameters.
    xnn_qu8_add_minmax_params params;
    struct xnn_quantization_params a_quantization = {a_zero_point(), a_scale()};
    struct xnn_quantization_params b_quantization = {b_zero_point(), b_scale()};
    struct xnn_quantization_params y_quantization = {y_zero_point(), y_scale()};
    init_params(&params, &a_quantization, &b_quantization, &y_quantization);

    // Compute reference results.
    for (size_t i = 0; i < batch_size(); i++) {
      y_fp[i] = static_cast<float>(y_zero_point()) +
                static_cast<float>(static_cast<int32_t>(a_data[i]) -
                                   static_cast<int32_t>(a_zero_point())) *
                    (a_scale() / y_scale()) +
                static_cast<float>(static_cast<int32_t>(b_data[i * stride_b]) -
                                   static_cast<int32_t>(b_zero_point())) *
                    (b_scale() / y_scale());
      y_fp[i] = std::min<float>(y_fp[i], static_cast<float>(UINT8_MAX));
      y_fp[i] = std::max<float>(y_fp[i], static_cast<float>(0));
      y_ref[i] = xnn_qu8_quantize_add(a_data[i], b_data[i * stride_b], params);
    }

    // Call optimized micro-kernel.
    vadd_minmax(batch_size(), a_data, b_data, y.data(), &params);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      ASSERT_NEAR(static_cast<float>(static_cast<int32_t>(y[i])), y_fp[i], 1.0f)
          << "at element " << i << " / " << batch_size();
      EXPECT_EQ(static_cast<uint32_t>(y_ref[i]), static_cast<uint32_t>(y[i]))
          << "at element " << i << " / " << batch_size();
    }
  }
}

void VBinaryMicrokernelTester::Test(
    xnn_qu8_vmul_minmax_ukernel_fn vmul_minmax,
    xnn_init_qu8_mul_minmax_params_fn init_params) const {
  xnnpack::ReplicableRandomDevice rng;
  auto u8rng = [&rng]() {
    return std::uniform_int_distribution<uint32_t>(
        0, std::numeric_limits<uint8_t>::max())(rng);
  };

  xnnpack::Buffer<uint8_t> a(batch_size(), xnnpack::XnnExtraBytes);
  xnnpack::Buffer<uint8_t> b(broadcast_b() ? 1 : batch_size(),
                             xnnpack::XnnExtraBytes);
  xnnpack::Buffer<uint8_t> y(
      batch_size(),
      xnnpack::PaddingBytes{inplace_a() || inplace_b() ? XNN_EXTRA_BYTES : 0});
  xnnpack::Buffer<float> y_fp(batch_size());
  xnnpack::Buffer<uint8_t> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    if (!inplace_a()) {
      std::generate(a.begin(), a.end(), [&]() { return u8rng(); });
    }
    if (!inplace_b()) {
      std::generate(b.begin(), b.end(), [&]() { return u8rng(); });
    }
    if (inplace_a() || inplace_b()) {
      std::generate(y.begin(), y.end(), [&]() { return u8rng(); });
    }
    const uint8_t* a_data = inplace_a() ? y.data() : a.data();
    const uint8_t* b_data = inplace_b() ? y.data() : b.data();
    const size_t stride_b = broadcast_b() ? 0 : 1;

    // Prepare parameters.
    const float product_scale = a_scale() * b_scale();
    const float product_output_scale = product_scale / y_scale();
    xnn_qu8_mul_minmax_params params;
    struct xnn_quantization_params a_quantization = {a_zero_point(), a_scale()};
    struct xnn_quantization_params b_quantization = {b_zero_point(), b_scale()};
    struct xnn_quantization_params y_quantization = {y_zero_point(), y_scale()};
    init_params(&params, &a_quantization, &b_quantization, &y_quantization);

    // Compute reference results.
    for (size_t i = 0; i < batch_size(); i++) {
      const int32_t acc = (static_cast<int32_t>(a_data[i]) -
                           static_cast<int32_t>(a_zero_point())) *
                          (static_cast<int32_t>(b_data[i * stride_b]) -
                           static_cast<int32_t>(b_zero_point()));
      y_fp[i] = static_cast<float>(y_zero_point()) +
                product_output_scale * static_cast<float>(acc);
      y_fp[i] = std::min<float>(y_fp[i], static_cast<float>(UINT8_MAX));
      y_fp[i] = std::max<float>(y_fp[i], static_cast<float>(0));
      y_ref[i] = xnn_qu8_requantize_fp32(acc, product_output_scale,
                                         y_zero_point(), 0, UINT8_MAX);
    }

    // Call optimized micro-kernel.
    vmul_minmax(batch_size(), a_data, b_data, y.data(), &params);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      ASSERT_NEAR(static_cast<float>(static_cast<int32_t>(y[i])), y_fp[i], 1.0f)
          << "at element " << i << " / " << batch_size();
      ASSERT_NEAR(static_cast<uint32_t>(y[i]), static_cast<uint32_t>(y_ref[i]),
                  1)
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

  xnnpack::Buffer<int8_t> a(batch_size(), xnnpack::XnnExtraBytes);
  xnnpack::Buffer<int8_t> b(batch_size(), xnnpack::XnnExtraBytes);
  xnnpack::Buffer<int8_t> y(
      batch_size(),
      xnnpack::PaddingBytes{inplace_a() || inplace_b() ? XNN_EXTRA_BYTES : 0});
  xnnpack::Buffer<float> y_fp(batch_size());
  xnnpack::Buffer<int8_t> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    if (!inplace_a()) {
      std::generate(a.begin(), a.end(), [&]() { return i8rng(); });
    }
    if (!inplace_b()) {
      std::generate(b.begin(), b.end(), [&]() { return i8rng(); });
    }
    if (inplace_a() || inplace_b()) {
      std::generate(y.begin(), y.end(), [&]() { return i8rng(); });
    }
    const int8_t* a_data = inplace_a() ? y.data() : a.data();
    const int8_t* b_data = inplace_b() ? y.data() : b.data();
    const size_t stride_b = broadcast_b() ? 0 : 1;

    // Prepare parameters.
    xnn_qs8_add_minmax_params params;
    struct xnn_quantization_params a_quantization = {a_zero_point() - 0x80,
                                                     a_scale()};
    struct xnn_quantization_params b_quantization = {b_zero_point() - 0x80,
                                                     b_scale()};
    struct xnn_quantization_params y_quantization = {y_zero_point() - 0x80,
                                                     y_scale()};
    init_params(&params, &a_quantization, &b_quantization, &y_quantization);

    // Compute reference results.
    for (size_t i = 0; i < batch_size(); i++) {
      y_fp[i] =
          static_cast<float>(static_cast<int32_t>(y_zero_point() - 0x80)) +
          static_cast<float>(static_cast<int32_t>(a_data[i]) -
                             static_cast<int32_t>(a_zero_point() - 0x80)) *
              (a_scale() / y_scale()) +
          static_cast<float>(static_cast<int32_t>(b_data[i * stride_b]) -
                             static_cast<int32_t>(b_zero_point() - 0x80)) *
              (b_scale() / y_scale());
      y_fp[i] = std::min<float>(y_fp[i], static_cast<float>(INT8_MAX));
      y_fp[i] = std::max<float>(y_fp[i], static_cast<float>(INT8_MIN));
      y_ref[i] = xnn_qs8_quantize_add(a_data[i], b_data[i * stride_b], params);
    }

    // Call optimized micro-kernel.
    vadd_minmax(batch_size(), a_data, b_data, y.data(), &params);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      EXPECT_EQ(static_cast<int32_t>(y_ref[i]), static_cast<int32_t>(y[i]))
          << "at element " << i << " / " << batch_size();
      ASSERT_NEAR(static_cast<float>(static_cast<int32_t>(y[i])), y_fp[i], 1.0f)
          << "at element " << i << " / " << batch_size();
    }
  }
}

void VBinaryMicrokernelTester::Test(
    xnn_qs8_vmul_minmax_ukernel_fn vmul_minmax,
    xnn_init_qs8_mul_minmax_params_fn init_params) const {
  xnnpack::ReplicableRandomDevice rng;
  auto i8rng = [&rng]() {
    return std::uniform_int_distribution<int32_t>(
        std::numeric_limits<int8_t>::min(),
        std::numeric_limits<int8_t>::max())(rng);
  };

  xnnpack::Buffer<int8_t> a(batch_size(), xnnpack::XnnExtraBytes);
  xnnpack::Buffer<int8_t> b(batch_size(), xnnpack::XnnExtraBytes);
  xnnpack::Buffer<int8_t> y(
      batch_size(),
      xnnpack::PaddingBytes{inplace_a() || inplace_b() ? XNN_EXTRA_BYTES : 0});
  xnnpack::Buffer<float> y_fp(batch_size());
  xnnpack::Buffer<int8_t> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    if (!inplace_a()) {
      std::generate(a.begin(), a.end(), [&]() { return i8rng(); });
    }
    if (!inplace_b()) {
      std::generate(b.begin(), b.end(), [&]() { return i8rng(); });
    }
    if (inplace_a() || inplace_b()) {
      std::generate(y.begin(), y.end(), [&]() { return i8rng(); });
    }
    const int8_t* a_data = inplace_a() ? y.data() : a.data();
    const int8_t* b_data = inplace_b() ? y.data() : b.data();
    const size_t stride_b = broadcast_b() ? 0 : 1;

    // Prepare parameters.
    xnn_qs8_mul_minmax_params params;
    struct xnn_quantization_params a_quantization = {a_zero_point() - 0x80,
                                                     a_scale()};
    struct xnn_quantization_params b_quantization = {b_zero_point() - 0x80,
                                                     b_scale()};
    struct xnn_quantization_params y_quantization = {y_zero_point() - 0x80,
                                                     y_scale()};
    init_params(&params, &a_quantization, &b_quantization, &y_quantization);

    // Compute reference results.
    const float product_scale = a_scale() * b_scale();
    const float product_output_scale = product_scale / y_scale();
    EXPECT_GE(product_output_scale, 0x1.0p-32f);
    for (size_t i = 0; i < batch_size(); i++) {
      const int32_t acc = (static_cast<int32_t>(a_data[i]) -
                           static_cast<int32_t>(a_zero_point() - 0x80)) *
                          (static_cast<int32_t>(b_data[i * stride_b]) -
                           static_cast<int32_t>(b_zero_point() - 0x80));
      y_fp[i] = static_cast<float>(y_zero_point() - 0x80) +
                product_output_scale * static_cast<float>(acc);
      y_fp[i] = std::min<float>(y_fp[i], static_cast<float>(INT8_MAX));
      y_fp[i] = std::max<float>(y_fp[i], static_cast<float>(INT8_MIN));
      y_ref[i] = xnn_qs8_requantize_fp32(
          acc, product_output_scale, static_cast<int8_t>(y_zero_point() - 0x80),
          INT8_MIN, INT8_MAX);
    }

    // Call optimized micro-kernel.
    vmul_minmax(batch_size(), a_data, b_data, y.data(), &params);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      ASSERT_NEAR(static_cast<int32_t>(y_ref[i]), static_cast<int32_t>(y[i]), 1)
          << "at element " << i << " / " << batch_size();
      ASSERT_NEAR(static_cast<float>(static_cast<int32_t>(y[i])), y_fp[i], 1.0f)
          << "at element " << i << " / " << batch_size();
    }
  }
}

void VBinaryMicrokernelTester::Test(
    xnn_qs8_vprelu_ukernel_fn vprelu, OpType op_type,
    xnn_init_qs8_vprelu_params_fn init_params) const {
  xnnpack::ReplicableRandomDevice rng;
  auto i8rng = [&rng]() {
    return std::uniform_int_distribution<int32_t>(
        std::numeric_limits<int8_t>::min(),
        std::numeric_limits<int8_t>::max())(rng);
  };

  xnnpack::Buffer<int8_t> a(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
  xnnpack::Buffer<int8_t> b(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
  xnnpack::Buffer<int8_t> y(
      batch_size() +
      (inplace_a() || inplace_b() ? XNN_EXTRA_BYTES / sizeof(int8_t) : 0));
  xnnpack::Buffer<float> y_fp(batch_size());
  xnnpack::Buffer<int8_t> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    if (!inplace_a()) {
      std::generate(a.begin(), a.end(), [&]() { return i8rng(); });
    }
    if (!inplace_b()) {
      std::generate(b.begin(), b.end(), [&]() { return i8rng(); });
    }
    if (inplace_a() || inplace_b()) {
      std::generate(y.begin(), y.end(), [&]() { return i8rng(); });
    }
    const int8_t* a_data = inplace_a() ? y.data() : a.data();
    const int8_t* b_data = inplace_b() ? y.data() : b.data();
    const size_t stride_b = broadcast_b() ? 0 : 1;

    // Prepare parameters.
    xnn_qs8_vprelu_scalar_params params;
    struct xnn_quantization_params a_quantization = {a_zero_point() - 0x80,
                                                     a_scale()};
    struct xnn_quantization_params b_quantization = {b_zero_point() - 0x80,
                                                     b_scale()};
    struct xnn_quantization_params y_quantization = {y_zero_point() - 0x80,
                                                     y_scale()};
    init_params(&params, &a_quantization, &b_quantization, &y_quantization);

    // Compute reference results.
    const float positive_multiplier = a_scale() / y_scale();
    const float rprelu_pos_multiplier = b_scale() / y_scale();
    const float negative_multiplier = (a_scale() * b_scale()) / y_scale();
    EXPECT_GE(positive_multiplier, 0x1.0p-32f);
    EXPECT_GE(negative_multiplier, 0x1.0p-32f);
    for (size_t i = 0; i < batch_size(); i++) {
      int32_t acc;
      float scale;
      const int32_t a_val = static_cast<int32_t>(a_data[i]) - static_cast<int32_t>(a_zero_point() - 0x80);
      const int32_t b_val = static_cast<int32_t>(b_data[i * stride_b]) - static_cast<int32_t>(b_zero_point() - 0x80);
      switch (op_type)
      {
      case OpType::Prelu:
        acc = (a_val < 0) ? a_val * b_val : a_val;
        scale = (a_val < 0) ? negative_multiplier : positive_multiplier;
        break;
      default:
        acc = (b_val < 0) ? a_val * b_val : b_val;
        scale = (b_val < 0) ? negative_multiplier : rprelu_pos_multiplier; 
        break;
      }
      y_fp[i] = static_cast<float>(y_zero_point() - 0x80) + scale * static_cast<float>(acc);
      y_fp[i] = std::min<float>(y_fp[i], static_cast<float>(INT8_MAX));
      y_fp[i] = std::max<float>(y_fp[i], static_cast<float>(INT8_MIN));
      y_ref[i] = xnn_qs8_requantize_fp32(
          acc, scale, static_cast<int8_t>(y_zero_point() - 0x80),
          INT8_MIN, INT8_MAX);
    }

    // Call optimized micro-kernel.
    vprelu(batch_size(), a_data, b_data, y.data(), &params);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      EXPECT_NEAR(static_cast<int32_t>(y_ref[i]), static_cast<int32_t>(y[i]), 1)
          << "at element " << i << " / " << batch_size();
      EXPECT_NEAR(static_cast<float>(static_cast<int32_t>(y[i])), y_fp[i], 1.0f)
          << "at element " << i << " / " << batch_size();
    }
  }
}



void VBinaryMicrokernelTester::Test(
    xnn_qu8_vprelu_ukernel_fn vprelu, OpType op_type,
    xnn_init_qu8_vprelu_params_fn init_params) const {
  xnnpack::ReplicableRandomDevice rng;
  auto u8rng = [&rng]() {
    return std::uniform_int_distribution<uint32_t>(
        0, std::numeric_limits<uint8_t>::max())(rng);
  };

  xnnpack::Buffer<uint8_t> a(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  xnnpack::Buffer<uint8_t> b(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  xnnpack::Buffer<uint8_t> y(
      batch_size() +
      (inplace_a() || inplace_b() ? XNN_EXTRA_BYTES / sizeof(uint8_t) : 0));
  xnnpack::Buffer<float> y_fp(batch_size());
  xnnpack::Buffer<uint8_t> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    if (!inplace_a()) {
      std::generate(a.begin(), a.end(), [&]() { return u8rng(); });
    }
    if (!inplace_b()) {
      std::generate(b.begin(), b.end(), [&]() { return u8rng(); });
    }
    if (inplace_a() || inplace_b()) {
      std::generate(y.begin(), y.end(), [&]() { return u8rng(); });
    }
    const uint8_t* a_data = inplace_a() ? y.data() : a.data();
    const uint8_t* b_data = inplace_b() ? y.data() : b.data();
    const size_t stride_b = broadcast_b() ? 0 : 1;

    // Prepare parameters.
    xnn_qs8_vprelu_scalar_params params;
    struct xnn_quantization_params a_quantization = {a_zero_point(), a_scale()};
    struct xnn_quantization_params b_quantization = {b_zero_point(), b_scale()};
    struct xnn_quantization_params y_quantization = {y_zero_point(), y_scale()};
    init_params(&params, &a_quantization, &b_quantization, &y_quantization);

    // Compute reference results.
    const float positive_multiplier = a_scale() / y_scale();
    const float rprelu_pos_multiplier = b_scale() / y_scale();
    const float negative_multiplier = (a_scale() * b_scale()) / y_scale();
   
    for (size_t i = 0; i < batch_size(); i++) {
      int32_t acc;
      float scale;
      const int32_t a_val = static_cast<int32_t>(a_data[i]) - static_cast<int32_t>(a_zero_point());
      const int32_t b_val = static_cast<int32_t>(b_data[i * stride_b]) - static_cast<int32_t>(b_zero_point());
      switch (op_type)
      {
      case OpType::Prelu:
        acc = (a_val < 0) ? a_val * b_val : a_val;
        scale = (a_val < 0) ? negative_multiplier : positive_multiplier;
        break;
      default:
        acc = (b_val < 0) ? a_val * b_val : b_val;
        scale = (b_val < 0) ? negative_multiplier : rprelu_pos_multiplier; 
        break;
      }
      y_fp[i] = static_cast<float>(y_zero_point()) + scale * static_cast<float>(acc);
      y_fp[i] = std::min<float>(y_fp[i], static_cast<float>(UINT8_MAX));
      y_fp[i] = std::max<float>(y_fp[i], static_cast<float>(0));
      y_ref[i] = xnn_qu8_requantize_fp32(
          acc, scale, static_cast<uint8_t>(y_zero_point()),
          0, UINT8_MAX);
    }

    // Call optimized micro-kernel.
    vprelu(batch_size(), a_data, b_data, y.data(), &params);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      EXPECT_NEAR(static_cast<uint32_t>(y_ref[i]), static_cast<uint32_t>(y[i]), 1)
          << "at element " << i << " / " << batch_size();
      EXPECT_NEAR(static_cast<float>(static_cast<int32_t>(y[i])), y_fp[i], 1.0f)
          << "at element " << i << " / " << batch_size();
    }
  }
}


