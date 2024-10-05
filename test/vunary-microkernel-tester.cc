// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "vunary-microkernel-tester.h"

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

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "replicable_random_device.h"

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
#endif

void VUnaryMicrokernelTester::Test(xnn_f32_vrelu_ukernel_fn vrelu,
                                   xnn_init_f32_relu_params_fn, Default) const {
  Test<float>(
      vrelu, [](xnn_f32_relu_params*) { return nullptr; },
      [](float x) { return std::max(x, 0.0f); }, TolExact, -1.0f, 1.0f);
}

void VUnaryMicrokernelTester::Test(xnn_bf16_vabs_ukernel_fn vabs,
                                   xnn_init_bf16_default_params_fn init_params,
                                   Abs) const {
  Test<xnn_bfloat16>(
      vabs, InitParamsWrapper(init_params), [](float x) { return std::abs(x); },
      TolExact16, -1.0f, 1.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f16_vabs_ukernel_fn vabs,
                                   xnn_init_f16_default_params_fn init_params,
                                   Abs) const {
  Test<xnn_float16>(
      vabs, InitParamsWrapper(init_params), [](float x) { return std::abs(x); },
      TolExact16, -1.0f, 1.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f32_vabs_ukernel_fn vabs,
                                   xnn_init_f32_default_params_fn init_params,
                                   Abs) const {
  Test<float>(
      vabs, InitParamsWrapper(init_params), [](float x) { return std::abs(x); },
      TolExact, -1.0f, 1.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f32_vclamp_ukernel_fn vclamp,
                                   xnn_init_f32_minmax_params_fn init_params,
                                   Default) const {
  Test<float>(
      vclamp,
      InitParamsWrapper(init_params, static_cast<float>(qmin()),
                        static_cast<float>(qmax())),
      [this](float x) {
        return std::max(std::min(x, static_cast<float>(qmax())),
                        static_cast<float>(qmin()));
      },
      TolExact, 0.0f, 255.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f16_velu_ukernel_fn velu,
                                   xnn_init_f16_elu_params_fn init_params,
                                   Default) const {
  Test<xnn_float16>(
      velu,
      InitParamsWrapper(init_params, 
                        xnn_float16(prescale()),
                        xnn_float16(alpha()),
                        xnn_float16(beta())),
      [this](float x) {
        return std::signbit(x) ? alpha() * std::expm1(x * prescale())
                               : x * beta();
      },
      TolMixed(1.0e-4f, 5.0e-3f), -9.0f, 9.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f32_velu_ukernel_fn velu,
                                   xnn_init_f32_elu_params_fn init_params,
                                   Default) const {
  Test<float>(
      velu, InitParamsWrapper(init_params, prescale(), alpha(), beta()),
      [this](float x) {
        return std::signbit(x)
                   ? alpha() * std::expm1(static_cast<double>(x) * prescale())
                   : static_cast<double>(x) * beta();
      },
      TolMixed(5.0e-6f, 1.0e-5f), -20.0f, 20.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f32_vgelu_ukernel_fn vgelu,
                                   xnn_init_f32_default_params_fn init_params,
                                   Gelu) const {
  Test<float>(
      vgelu, InitParamsWrapper(init_params),
      [](float x) { return x * 0.5f * (1.0f + std::erf(x * M_SQRT1_2)); },
      TolMixed(10 * std::numeric_limits<float>::epsilon(),
               5 * std::numeric_limits<float>::epsilon()),
      -10.0f, 10.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f16_vhswish_ukernel_fn vhswish,
                                   xnn_init_f16_hswish_params_fn init_params,
                                   Default) const {
  Test<xnn_float16>(
      vhswish, InitParamsWrapper(init_params),
      [](float x) {
        return (x / 6.0f) * std::max(std::min(x + 3.0f, 6.0f), 0.0f);
      },
      TolMixed(1.0e-3f, 1.0e-2f), -4.0f, 4.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f32_vhswish_ukernel_fn vhswish,
                                   xnn_init_f32_hswish_params_fn init_params,
                                   Default) const {
  Test<float>(
      vhswish, InitParamsWrapper(init_params),
      [](float x) {
        return (x / 6.0f) * std::max(std::min(x + 3.0f, 6.0f), 0.0f);
      },
      TolMixed(5.0e-6f, 1.0e-5f), -4.0f, 4.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f16_vlrelu_ukernel_fn vlrelu,
                                   xnn_init_f16_lrelu_params_fn init_params,
                                   Default) const {
  const xnn_float16 slope_as_half = slope();
  const float slope_as_float = slope_as_half;
  Test<xnn_float16>(
      vlrelu, InitParamsWrapper(init_params, slope_as_half),
      [slope_as_float](float x) {
        return std::signbit(x) ? x * slope_as_float : x;
      },
      TolMixed(1.0e-4f, 1.0e-3f), -125.0f, 125.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f32_vlrelu_ukernel_fn vlrelu,
                                   xnn_init_f32_lrelu_params_fn init_params,
                                   Default) const {
  Test<float>(
      vlrelu, InitParamsWrapper(init_params, slope()),
      [this](float x) { return std::signbit(x) ? x * slope() : x; }, TolExact,
      -125.0f, 125.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f16_vneg_ukernel_fn vneg,
                                   xnn_init_f16_default_params_fn init_params,
                                   Neg) const {
  Test<xnn_float16>(
      vneg, InitParamsWrapper(init_params), [](float x) { return -x; },
      TolExact16, -1.0f, 1.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f32_vneg_ukernel_fn vneg,
                                   xnn_init_f32_default_params_fn init_params,
                                   Neg) const {
  Test<float>(
      vneg, InitParamsWrapper(init_params), [](float x) { return -x; },
      TolExact, -1.0f, 1.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f16_vround_ukernel_fn vrnd,
                                   OpType op_type,
                                   xnn_init_f16_rnd_params_fn init_params,
                                   Default) const {
  Test<xnn_float16>(
      vrnd, InitParamsWrapper(init_params),
      [op_type](float x) -> float {
        switch (op_type) {
          case OpType::RoundToNearestEven:
            return std::nearbyint(x);
          case OpType::RoundTowardsZero:
            return std::trunc(x);
          case OpType::RoundUp:
            return std::ceil(x);
          case OpType::RoundDown:
            return std::floor(x);
          default:
            []() { GTEST_FAIL() << "Unexpected operation type"; }();
            return 0.0f;
        }
      },
      TolExact16, -5.0f, 5.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f32_vround_ukernel_fn vrnd,
                                   OpType op_type,
                                   xnn_init_f32_rnd_params_fn init_params,
                                   Default) const {
  Test<float>(
      vrnd, InitParamsWrapper(init_params),
      [op_type](float x) -> float {
        switch (op_type) {
          case OpType::RoundToNearestEven:
            return std::nearbyint(x);
          case OpType::RoundTowardsZero:
            return std::trunc(x);
          case OpType::RoundUp:
            return std::ceil(x);
          case OpType::RoundDown:
            return std::floor(x);
          default:
            []() { GTEST_FAIL() << "Unexpected operation type"; }();
            return 0.0f;
        }
      },
      TolExact, -5.0f, 5.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f16_vsigmoid_ukernel_fn vsigmoid,
                                   xnn_init_f16_sigmoid_params_fn init_params,
                                   Default) const {
  Test<xnn_float16>(
      vsigmoid, InitParamsWrapper(init_params),
      [](float x) {
        const float e = std::exp(x);
        return e / (1.0f + e);
      },
      TolMixed(1.0e-4f, 5.0e-3f), -25.0f, 25.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f32_vsigmoid_ukernel_fn vsigmoid,
                                   xnn_init_f32_sigmoid_params_fn init_params,
                                   Default) const {
  Test<float>(
      vsigmoid, InitParamsWrapper(init_params),
      [](float x) {
        const double e = std::exp(static_cast<double>(x));
        return e / (1.0 + e);
      },
      TolMixed(5.0e-6f, 1.0e-5f), -125.0f, 125.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f16_vsqr_ukernel_fn vsqr,
                                   xnn_init_f16_default_params_fn init_params,
                                   Sqr) const {
  Test<xnn_float16>(
      vsqr, InitParamsWrapper(init_params), [](float x) { return x * x; },
      TolMixed(1.0e-4f, 5.0e-3f), -10.0f, 10.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f32_vsqr_ukernel_fn vsqr,
                                   xnn_init_f32_default_params_fn init_params,
                                   Sqr) const {
  Test<float>(
      vsqr, InitParamsWrapper(init_params), [](float x) { return x * x; },
      TolExact, -10.0f, 10.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f16_vsqrt_ukernel_fn vsqrt,
                                   xnn_init_f16_sqrt_params_fn init_params,
                                   Default) const {
  Test<xnn_float16>(
      vsqrt, InitParamsWrapper(init_params),
      [](float x) { return std::sqrt(x); }, TolMixed(1.0e-4f, 5.0e-3f), 0.001f,
      10.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f32_vexp_ukernel_fn vexp,
                                   xnn_init_f32_default_params_fn init_params,
                                   Exp) const {
  Test<float>(
      vexp, InitParamsWrapper(init_params), [](float x) { return std::exp(x); },
      TolMixed(2 * std::numeric_limits<float>::epsilon(),
               6 * std::numeric_limits<float>::epsilon()),
      0.0f, 10.0f);
}
void VUnaryMicrokernelTester::Test(xnn_f32_vlog_ukernel_fn vlog,
                                   xnn_init_f32_default_params_fn init_params,
                                   Log) const {
  Test<float>(
      vlog, InitParamsWrapper(init_params), [](float x) { return std::log(x); },
      TolMixed(2 * std::numeric_limits<float>::epsilon(),
               6 * std::numeric_limits<float>::epsilon()),
      0.0f, 10.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f32_vsqrt_ukernel_fn vsqrt,
                                   xnn_init_f32_sqrt_params_fn init_params,
                                   Default) const {
  Test<float>(
      vsqrt, InitParamsWrapper(init_params),
      [](float x) { return std::sqrt(x); },
      TolRelative(2.5f * std::numeric_limits<float>::epsilon()), 0.0f, 10.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f16_vrsqrt_ukernel_fn vrsqrt,
                                   xnn_init_f16_rsqrt_params_fn init_params,
                                   Default) const {
  Test<xnn_float16>(
      vrsqrt, InitParamsWrapper(init_params),
      [](float x) { return 1.0f / std::sqrt(x); }, TolMixed(1.0e-4f, 5.0e-3f),
      1.0e-4f, 10.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f32_vrsqrt_ukernel_fn vrsqrt,
                                   xnn_init_f32_rsqrt_params_fn init_params,
                                   Default) const {
  Test<float>(
      vrsqrt, InitParamsWrapper(init_params),
      [](float x) { return 1.0f / std::sqrt(x); },
      TolRelative(4 * std::numeric_limits<float>::epsilon()),
      std::numeric_limits<float>::epsilon(), 10.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f16_vtanh_ukernel_fn vtanh,
                                   xnn_init_f16_tanh_params_fn init_params,
                                   Default) const {
  Test<xnn_float16>(
      vtanh, InitParamsWrapper(init_params),
      [](float x) { return std::tanh(x); },
      TolMixed(/*abs_tol=*/1.0e-4f, /*rel_tol=*/5.0e-3f), -5.0f, 5.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f32_vtanh_ukernel_fn vtanh,
                                   xnn_init_f32_tanh_params_fn init_params,
                                   Default) const {
  Test<float>(
      vtanh, InitParamsWrapper(init_params),
      [](float x) { return std::tanh(x); },
      TolRelative(4.0f * std::numeric_limits<float>::epsilon()),  // 4 ULP.
      -10.0f, 10.0f);
}

void VUnaryMicrokernelTester::Test(xnn_f16_vclamp_ukernel_fn vclamp,
                                   xnn_init_f16_minmax_params_fn init_params,
                                   Default) const {
  Test<xnn_float16>(
      vclamp,
      InitParamsWrapper(init_params,
                        xnn_float16(qmin()),
                        xnn_float16(qmax())),
      [this](float x) {
        return std::max(std::min(x, static_cast<float>(qmax())),
                        static_cast<float>(qmin()));
      },
      TolExact16, 0.0f, 255.0f);
}

void VUnaryMicrokernelTester::Test(xnn_s8_vclamp_ukernel_fn vclamp,
                                   xnn_init_s8_minmax_params_fn init_params,
                                   Default) const {
  xnnpack::ReplicableRandomDevice rng;
  auto i8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             std::numeric_limits<int8_t>::min(),
                             std::numeric_limits<int8_t>::max()),
                         std::ref(rng));

  std::vector<int8_t> x(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<int8_t> y(batch_size() +
                        (inplace() ? XNN_EXTRA_BYTES / sizeof(int8_t) : 0));
  std::vector<int8_t> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(x.begin(), x.end(), std::ref(i8rng));
    if (inplace()) {
      std::copy(x.cbegin(), x.cend(), y.begin());
    } else {
      std::fill(y.begin(), y.end(), INT8_C(0xA5));
    }
    const int8_t* x_data = inplace() ? y.data() : x.data();

    // Compute reference results.
    for (size_t i = 0; i < batch_size(); i++) {
      y_ref[i] =
          std::min(std::max(x_data[i], static_cast<int8_t>(qmin() - 0x80)),
                   static_cast<int8_t>(qmax() - 0x80));
    }

    // Prepare parameters.
    struct xnn_s8_minmax_params params;
    init_params(&params, static_cast<int8_t>(qmin() - 0x80),
                static_cast<int8_t>(qmax() - 0x80));

    // Call optimized micro-kernel.
    vclamp(batch_size() * sizeof(int8_t), x_data, y.data(), &params);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      EXPECT_EQ(static_cast<int32_t>(y_ref[i]), static_cast<int32_t>(y[i]))
          << "at " << i << " / " << batch_size() << ", x[" << i
          << "] = " << int32_t(x[i]);
    }
  }
}

void VUnaryMicrokernelTester::Test(xnn_u8_vclamp_ukernel_fn vclamp,
                                   xnn_init_u8_minmax_params_fn init_params,
                                   Default) const {
  xnnpack::ReplicableRandomDevice rng;
  auto u8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             0, std::numeric_limits<uint8_t>::max()),
                         std::ref(rng));

  std::vector<uint8_t> x(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<uint8_t> y(batch_size() +
                         (inplace() ? XNN_EXTRA_BYTES / sizeof(uint8_t) : 0));
  std::vector<uint8_t> y_ref(batch_size());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(x.begin(), x.end(), std::ref(u8rng));
    if (inplace()) {
      std::copy(x.cbegin(), x.cend(), y.begin());
    } else {
      std::fill(y.begin(), y.end(), UINT8_C(0xA5));
    }
    const uint8_t* x_data = inplace() ? y.data() : x.data();

    // Compute reference results.
    for (size_t i = 0; i < batch_size(); i++) {
      y_ref[i] = std::min(std::max(x_data[i], qmin()), qmax());
    }

    // Prepare parameters.
    struct xnn_u8_minmax_params params;
    init_params(&params, qmin(), qmax());

    // Call optimized micro-kernel.
    vclamp(batch_size() * sizeof(uint8_t), x_data, y.data(), &params);

    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      EXPECT_EQ(static_cast<uint32_t>(y_ref[i]), static_cast<uint32_t>(y[i]))
          << "at " << i << " / " << batch_size() << ", x[" << i
          << "] = " << uint32_t(x[i]);
    }
  }
}
