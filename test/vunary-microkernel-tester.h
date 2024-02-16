// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <xnnpack.h>
#include <xnnpack/common.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/microparams.h>

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

#include <fp16/fp16.h>
#include <gtest/gtest.h>

#if XNN_PLATFORM_JIT
  #include <xnnpack/memory.h>
#endif

class VUnaryMicrokernelTester {
 public:
  enum class OpType {
    ReLU,
    RoundToNearestEven,
    RoundTowardsZero,
    RoundUp,
    RoundDown,
  };

  inline VUnaryMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline VUnaryMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  inline bool inplace() const {
    return this->inplace_;
  }

  inline VUnaryMicrokernelTester& slope(float slope) {
    this->slope_ = slope;
    return *this;
  }

  inline float slope() const {
    return this->slope_;
  }

  inline VUnaryMicrokernelTester& prescale(float prescale) {
    this->prescale_ = prescale;
    return *this;
  }

  inline float prescale() const {
    return this->prescale_;
  }

  inline VUnaryMicrokernelTester& alpha(float alpha) {
    this->alpha_ = alpha;
    return *this;
  }

  inline float alpha() const {
    return this->alpha_;
  }

  inline VUnaryMicrokernelTester& beta(float beta) {
    this->beta_ = beta;
    return *this;
  }

  inline float beta() const {
    return this->beta_;
  }

  inline VUnaryMicrokernelTester& shift(uint32_t shift) {
    this->shift_ = shift;
    return *this;
  }

  inline uint32_t shift() const {
    return this->shift_;
  }

  inline VUnaryMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline VUnaryMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline VUnaryMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  // Generic test function for `fp32` `vunary` kernels.
  //
  // The function is templated on the type of the kernel parameters and takes
  // the following arguments:
  //
  //  * `init_params`: A function that populates a given parameters data
  //    structure or returns `nullptr` if there is no default initialization.
  //  * `ref`: A function that computes the reference result for an input `x`.
  //  * `tol`: A function that computes the absolute tolerance for a reference
  //    result `y_ref`.
  //  * `range_min`, `range_max`: Limits for the range of input values.
  template <typename UKernelParamsType, typename InitParamsFunc,
            typename ReferenceFunc, typename ToleranceFunc>
  void TestFP32(void (*ukernel)(size_t, const float*, float*,
                                const UKernelParamsType*),
                InitParamsFunc init_params, ReferenceFunc ref,
                ToleranceFunc tol, float range_min, float range_max) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(range_min, range_max);

    std::vector<float> x(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(batch_size() +
                         (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<float> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      if (inplace()) {
        std::generate(y.begin(), y.end(), [&]() { return f32dist(rng); });
      } else {
        std::generate(x.begin(), x.end(), [&]() { return f32dist(rng); });
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = ref(x_data[i]);
      }

      // Initialize the params.
      UKernelParamsType params;
      const UKernelParamsType* params_ptr = init_params(&params);

      // Call optimized micro-kernel.
      ukernel(batch_size() * sizeof(float), x_data, y.data(), params_ptr);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_NEAR(y[i], y_ref[i], tol(y_ref[i]))
            << "at " << i << " / " << batch_size() << ", x[" << i
            << "] = " << x[i];
      }
    }
  }

  // Generic test function for `fp16` `vunary` kernels.
  //
  // The function is templated on the type of the kernel parameters and takes
  // the following arguments:
  //
  //  * `init_params`: A function that populates a given parameters data
  //    structure or returns `nullptr` if there is no default initialization.
  //  * `ref`: A function that computes the reference result for an input `x` of
  //    type `float`, converted from the actual `fp16` input.
  //  * `tol`: A function that computes the absolute tolerance for a reference
  //    result `y_ref` of type `float`. Note that the computed result `y` will
  //    be converted back to `float` for the comparison.
  //  * `range_min`, `range_max`: Limits for the range of input values.
  template <typename UKernelParamsType, typename InitParamsFunc,
            typename ReferenceFunc, typename ToleranceFunc>
  void TestFP16(void (*ukernel)(size_t, const void*, void*,
                                const UKernelParamsType*),
                InitParamsFunc init_params, ReferenceFunc ref,
                ToleranceFunc tol, float range_min, float range_max) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto distribution =
        std::uniform_real_distribution<float>(range_min, range_max);
    auto f16rng = [&]() {
      return fp16_ieee_from_fp32_value(distribution(rng));
    };

    std::vector<uint16_t> x(batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> y(
        batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(uint16_t) : 0));
    std::vector<float> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(f16rng));
      } else {
        std::generate(x.begin(), x.end(), std::ref(f16rng));
        std::fill(y.begin(), y.end(), UINT16_C(0x7E00) /* NaN */);
      }
      const uint16_t* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = ref(fp16_ieee_to_fp32_value(x_data[i]));
      }

      // Initialize the params.
      UKernelParamsType params;
      const UKernelParamsType* params_ptr = init_params(&params);

      // Call optimized micro-kernel.
      ukernel(batch_size() * sizeof(uint16_t), x_data, y.data(), params_ptr);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_NEAR(fp16_ieee_to_fp32_value(y[i]), y_ref[i], tol(y_ref[i]))
            << "at " << i << " / " << batch_size() << ", x[" << i
            << "] = " << fp16_ieee_to_fp32_value(x[i]);
      }
    }
  }

  // Wrapper that generate the `init_params` functions needed by `TestFP32` and
  // `TestFP16` from the microkernel parameter initializer pointers, for
  // different numbers of additional inputs.
  template <typename UKernelParamsType, typename... Ts>
  static std::function<UKernelParamsType*(UKernelParamsType*)>
  InitParamsWrapper(size_t (*init_params)(UKernelParamsType*, Ts...),
                    Ts... args) {
    return [=](UKernelParamsType* params) -> UKernelParamsType* {
      if (init_params != nullptr) {
        init_params(params, args...);
        return params;
      }
      return nullptr;
    };
  }

  // Tolerance functions for the `TestFP32` and `TestFP16` template functions.
  static float TolExact(float) { return 0.0f; }
  static float TolExact16(float y_ref) { return std::abs(y_ref) * 5.0e-4f; }
  static std::function<float(float)> TolMixed(float abs_tol, float rel_tol) {
    return [=](float y_ref) -> float {
      return std::max(abs_tol, std::abs(y_ref) * rel_tol);
    };
  }
  static std::function<float(float)> TolRelative(float rel_tol) {
    return [=](float y_ref) -> float { return std::abs(y_ref) * rel_tol; };
  }

  void Test(xnn_f32_vrelu_ukernel_fn vrelu) const {
    TestFP32(
        vrelu, [](xnn_f32_relu_params*) { return nullptr; },
        [](float x) { return std::max(x, 0.0f); }, TolExact, -1.0f, 1.0f);
  }

  void Test(xnn_f16_vabs_ukernel_fn vabs,
            xnn_init_f16_abs_params_fn init_params = nullptr) const {
    TestFP16(
        vabs, InitParamsWrapper(init_params),
        [](float x) { return std::abs(x); }, TolExact16, -1.0f, 1.0f);
  }

  void Test(xnn_f32_vabs_ukernel_fn vabs,
            xnn_init_f32_abs_params_fn init_params = nullptr) const {
    TestFP32(
        vabs, InitParamsWrapper(init_params),
        [](float x) { return std::abs(x); }, TolExact, -1.0f, 1.0f);
  }

  void Test(xnn_f32_vclamp_ukernel_fn vclamp,
            xnn_init_f32_minmax_params_fn init_params) const {
    TestFP32(
        vclamp,
        InitParamsWrapper(init_params, static_cast<float>(qmin()),
                          static_cast<float>(qmax())),
        [this](float x) {
          return std::max(std::min(x, static_cast<float>(qmax())),
                          static_cast<float>(qmin()));
        },
        TolExact, 0.0f, 255.0f);
  }

  void Test(xnn_f16_velu_ukernel_fn velu,
            xnn_init_f16_elu_params_fn init_params) const {
    TestFP16(
        velu,
        InitParamsWrapper(init_params, fp16_ieee_from_fp32_value(prescale()),
                          fp16_ieee_from_fp32_value(alpha()),
                          fp16_ieee_from_fp32_value(beta())),
        [this](float x) {
          return std::signbit(x) ? alpha() * std::expm1(x * prescale())
                                 : x * beta();
        },
        TolMixed(1.0e-4f, 5.0e-3f), -9.0f, 9.0f);
  }

  void Test(xnn_f32_velu_ukernel_fn velu,
            xnn_init_f32_elu_params_fn init_params) const {
    TestFP32(
        velu, InitParamsWrapper(init_params, prescale(), alpha(), beta()),
        [this](float x) {
          return std::signbit(x)
                     ? alpha() * std::expm1(static_cast<double>(x) * prescale())
                     : static_cast<double>(x) * beta();
        },
        TolMixed(5.0e-6f, 1.0e-5f), -20.0f, 20.0f);
  }

  void Test(xnn_f16_vhswish_ukernel_fn vhswish,
            xnn_init_f16_hswish_params_fn init_params) const {
    TestFP16(
        vhswish, InitParamsWrapper(init_params),
        [](float x) {
          return (x / 6.0f) * std::max(std::min(x + 3.0f, 6.0f), 0.0f);
        },
        TolMixed(1.0e-3f, 1.0e-2f), -4.0f, 4.0f);
  }

  void Test(xnn_f32_vhswish_ukernel_fn vhswish,
            xnn_init_f32_hswish_params_fn init_params) const {
    TestFP32(
        vhswish, InitParamsWrapper(init_params),
        [](float x) {
          return (x / 6.0f) * std::max(std::min(x + 3.0f, 6.0f), 0.0f);
        },
        TolMixed(5.0e-6f, 1.0e-5f), -4.0f, 4.0f);
  }

  void Test(xnn_f16_vlrelu_ukernel_fn vlrelu,
            xnn_init_f16_lrelu_params_fn init_params) const {
    const uint16_t slope_as_half = fp16_ieee_from_fp32_value(slope());
    const float slope_as_float = fp16_ieee_to_fp32_value(slope_as_half);
    TestFP16(
        vlrelu, InitParamsWrapper(init_params, slope_as_half),
        [slope_as_float](float x) {
          return std::signbit(x) ? x * slope_as_float : x;
        },
        TolMixed(1.0e-4f, 1.0e-3f), -125.0f, 125.0f);
  }

  void Test(xnn_f32_vlrelu_ukernel_fn vlrelu,
            xnn_init_f32_lrelu_params_fn init_params) const {
    TestFP32(
        vlrelu, InitParamsWrapper(init_params, slope()),
        [this](float x) { return std::signbit(x) ? x * slope() : x; }, TolExact,
        -125.0f, 125.0f);
  }

  void Test(xnn_f16_vneg_ukernel_fn vneg,
            xnn_init_f16_neg_params_fn init_params = nullptr) const {
    TestFP16(
        vneg, InitParamsWrapper(init_params), [](float x) { return -x; },
        TolExact16, -1.0f, 1.0f);
  }

  void Test(xnn_f32_vneg_ukernel_fn vneg,
            xnn_init_f32_neg_params_fn init_params = nullptr) const {
    TestFP32(
        vneg, InitParamsWrapper(init_params), [](float x) { return -x; },
        TolExact, -1.0f, 1.0f);
  }

  void Test(xnn_f16_vround_ukernel_fn vrnd, OpType op_type,
            xnn_init_f16_rnd_params_fn init_params = nullptr) const {
    TestFP16(
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

  void Test(xnn_f32_vround_ukernel_fn vrnd, OpType op_type,
            xnn_init_f32_rnd_params_fn init_params = nullptr) const {
    TestFP32(
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

  void Test(xnn_f16_vsigmoid_ukernel_fn vsigmoid,
            xnn_init_f16_sigmoid_params_fn init_params) const {
    TestFP16(
        vsigmoid, InitParamsWrapper(init_params),
        [](float x) {
          const float e = std::exp(x);
          return e / (1.0f + e);
        },
        TolMixed(1.0e-4f, 5.0e-3f), -25.0f, 25.0f);
  }

  void Test(xnn_f32_vsigmoid_ukernel_fn vsigmoid,
            xnn_init_f32_sigmoid_params_fn init_params) const {
    TestFP32(
        vsigmoid, InitParamsWrapper(init_params),
        [](float x) {
          const double e = std::exp(static_cast<double>(x));
          return e / (1.0 + e);
        },
        TolMixed(5.0e-6f, 1.0e-5f), -125.0f, 125.0f);
  }

  void Test(xnn_f16_vsqr_ukernel_fn vsqr,
            xnn_init_f16_default_params_fn init_params = nullptr) const {
    TestFP16(
        vsqr, InitParamsWrapper(init_params), [](float x) { return x * x; },
        TolMixed(1.0e-4f, 5.0e-3f), -10.0f, 10.0f);
  }

  void Test(xnn_f32_vsqr_ukernel_fn vsqr,
            xnn_init_f32_default_params_fn init_params = nullptr) const {
    TestFP32(
        vsqr, InitParamsWrapper(init_params), [](float x) { return x * x; },
        TolExact, -10.0f, 10.0f);
  }

  void Test(xnn_f16_vsqrt_ukernel_fn vsqrt,
            xnn_init_f16_sqrt_params_fn init_params = nullptr) const {
    TestFP16(
        vsqrt, InitParamsWrapper(init_params),
        [](float x) { return std::sqrt(x); }, TolMixed(1.0e-4f, 5.0e-3f),
        0.001f, 10.0f);
  }

  void Test(xnn_f32_vsqrt_ukernel_fn vsqrt,
            xnn_init_f32_sqrt_params_fn init_params = nullptr) const {
    TestFP32(
        vsqrt, InitParamsWrapper(init_params),
        [](float x) { return std::sqrt(x); }, TolExact, 0.0f, 10.0f);
  }

  void Test(xnn_f32_vrsqrt_ukernel_fn vrsqrt,
            xnn_init_f32_rsqrt_params_fn init_params = nullptr) const {
    TestFP32(
        vrsqrt, InitParamsWrapper(init_params),
        [](float x) { return 1.0f / std::sqrt(x); },
        TolRelative(4 * std::numeric_limits<float>::epsilon()),
        std::numeric_limits<float>::epsilon(), 10.0f);
  }

  void Test(xnn_f16_vtanh_ukernel_fn vtanh,
            xnn_init_f16_tanh_params_fn init_params = nullptr) const {
    TestFP16(
        vtanh, InitParamsWrapper(init_params),
        [](float x) { return std::tanh(x); },
        TolMixed(/*abs_tol=*/1.0e-4f, /*rel_tol=*/5.0e-3f), -5.0f, 5.0f);
  }

  void Test(xnn_f32_vtanh_ukernel_fn vtanh,
            xnn_init_f32_tanh_params_fn init_params) const {
    TestFP32(
        vtanh, InitParamsWrapper(init_params),
        [](float x) { return std::tanh(x); },
        TolMixed(/*abs_tol=*/5.0e-6f, /*rel_tol=*/1.0e-5f), -10.0f, 10.0f);
  }

  void Test(xnn_f16_vclamp_ukernel_fn vclamp,
            xnn_init_f16_minmax_params_fn init_params) const {
    TestFP16(
        vclamp,
        InitParamsWrapper(
            init_params, fp16_ieee_from_fp32_value(static_cast<float>(qmin())),
            fp16_ieee_from_fp32_value(static_cast<float>(qmax()))),
        [this](float x) {
          return std::max(std::min(x, static_cast<float>(qmax())),
                          static_cast<float>(qmin()));
        },
        TolExact16, 0.0f, 255.0f);
  }

  void Test(xnn_s8_vclamp_ukernel_fn vclamp, xnn_init_s8_minmax_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto i8rng = std::bind(
      std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
      std::ref(rng));

    std::vector<int8_t> x(batch_size() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(int8_t) : 0));
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
        y_ref[i] = std::min(std::max(x_data[i], int8_t(qmin() - 0x80)), int8_t(qmax() - 0x80));
      }

      // Prepare parameters.
      union xnn_s8_minmax_params params;
      init_params(&params, int8_t(qmin() - 0x80), int8_t(qmax() - 0x80));

      // Call optimized micro-kernel.
      vclamp(batch_size() * sizeof(int8_t), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_EQ(int32_t(y_ref[i]), int32_t(y[i]))
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << int32_t(x[i]);
      }
    }
  }

  void Test(xnn_u8_vclamp_ukernel_fn vclamp, xnn_init_u8_minmax_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(
      std::uniform_int_distribution<int32_t>(0, std::numeric_limits<uint8_t>::max()), std::ref(rng));

    std::vector<uint8_t> x(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> y(batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(uint8_t) : 0));
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
      union xnn_u8_minmax_params params;
      init_params(&params, qmin(), qmax());

      // Call optimized micro-kernel.
      vclamp(batch_size() * sizeof(uint8_t), x_data, y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_EQ(uint32_t(y_ref[i]), uint32_t(y[i]))
          << "at " << i << " / " << batch_size() << ", x[" << i << "] = " << uint32_t(x[i]);
      }
    }
  }

  void Test(xnn_u64_u32_vsqrtshift_ukernel_fn vsqrtshift) const {
    ASSERT_FALSE(inplace());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u64rng = std::bind( std::uniform_int_distribution<uint64_t>(), std::ref(rng));

    std::vector<uint64_t> x(batch_size() + XNN_EXTRA_BYTES / sizeof(uint64_t));
    std::vector<uint32_t> y(batch_size());
    std::vector<uint32_t> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), std::ref(u64rng));
      std::fill(y.begin(), y.end(), UINT32_C(0xDEADBEEF));

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        const uint64_t x_value = x[i];
        uint32_t y_value = 0;
        // Match TFLM semantics, including bugs
        if (uint32_t(x_value) == x_value) {
          y_value = (uint32_t) std::lrint(std::sqrt(double(int64_t(uint64_t(x_value)))));
          y_value = std::min<uint32_t>(y_value, std::numeric_limits<uint16_t>::max());
        } else if (x_value != 0) {
          uint64_t y0 = x_value >> 1;
          uint64_t y1 = (y0 + x_value / y0) >> 1;
          do {
            y0 = y1;
            y1 = (y0 + x_value / y0) >> 1;
          } while (y1 < y0);

          // y0 is sqrt(x_value) rounded down, round up if needed
          if (int64_t(y0 * y0 + y0 - x_value) < 0) {
            y0 += 1;
          }
          y_value = static_cast<uint32_t>(std::min<uint64_t>(y0, std::numeric_limits<uint32_t>::max()));
        }
        y_ref[i] = y_value >> shift();
      }

      // Call optimized micro-kernel.
      vsqrtshift(batch_size() * sizeof(uint64_t), x.data(), y.data(), shift());

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        EXPECT_EQ(y_ref[i], y[i])
          << "at " << i << " / " << batch_size()
          << ", x[" << i << "]: " << x[i]
          << ", shift: " << shift();
      }
    }
  }

#if XNN_PLATFORM_JIT
  void Test(xnn_vrelu_generator_fn generator, size_t k_unroll, bool use_locals) const {
    xnn_code_buffer b;
    ASSERT_EQ(xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE), xnn_status_success);
    ASSERT_EQ(generator(&b, k_unroll, use_locals), xnn_status_success);
    ASSERT_EQ(xnn_finalize_code_memory(&b), xnn_status_success);
    auto kernel = (xnn_f32_vrelu_ukernel_fn)(xnn_first_function_ptr(&b));
    Test(kernel);
    xnn_release_code_memory(&b);
  }
#endif // XNN_PLATFORM_JIT

 private:
  size_t batch_size_ = 1;
  bool inplace_ = false;
  float slope_ = 0.5f;
  float prescale_ = 1.0f;
  float alpha_ = 1.0f;
  float beta_ = 1.0f;
  uint32_t shift_ = 1;
  uint8_t qmin_ = 0;
  uint8_t qmax_ = 255;
  size_t iterations_ = 15;
};
