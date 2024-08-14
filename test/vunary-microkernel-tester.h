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
#include <cstring>
#include <functional>
#include <ios>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "replicable_random_device.h"

class VUnaryMicrokernelTester {
 public:
  enum class OpType {
    ReLU,
    RoundToNearestEven,
    RoundTowardsZero,
    RoundUp,
    RoundDown,
  };

  VUnaryMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const { return this->batch_size_; }

  VUnaryMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  bool inplace() const { return this->inplace_; }

  VUnaryMicrokernelTester& slope(float slope) {
    this->slope_ = slope;
    return *this;
  }

  float slope() const { return this->slope_; }

  VUnaryMicrokernelTester& prescale(float prescale) {
    this->prescale_ = prescale;
    return *this;
  }

  float prescale() const { return this->prescale_; }

  VUnaryMicrokernelTester& alpha(float alpha) {
    this->alpha_ = alpha;
    return *this;
  }

  float alpha() const { return this->alpha_; }

  VUnaryMicrokernelTester& beta(float beta) {
    this->beta_ = beta;
    return *this;
  }

  float beta() const { return this->beta_; }

  VUnaryMicrokernelTester& shift(uint32_t shift) {
    this->shift_ = shift;
    return *this;
  }

  uint32_t shift() const { return this->shift_; }

  VUnaryMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  uint8_t qmin() const { return this->qmin_; }

  VUnaryMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  uint8_t qmax() const { return this->qmax_; }

  VUnaryMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const { return this->iterations_; }

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
  static std::function<float(float)> TolRelative(float rel_tol) {
    return [=](float y_ref) -> float {
      // Note that `y_ref * rel_tol`, i.e. the expected absolute difference,
      // may round differently than `y_ref * (1 + rel_tol) - y_ref`, i.e. the
      // effective absolute difference computed in `float`s. We therefore use
      // the latter form since it is the true difference between two `float`s
      // within the given relative tolerance.
      return std::abs(y_ref * (1.0f + rel_tol)) - std::abs(y_ref);
    };
  }
  static std::function<float(float)> TolMixed(float abs_tol, float rel_tol) {
    return [=](float y_ref) -> float {
      return std::max(abs_tol,
                      std::abs(y_ref) * (1.0f + rel_tol) - std::abs(y_ref));
    };
  }

  void Test(xnn_f32_vrelu_ukernel_fn vrelu) const;

  void TestAbs(xnn_bf16_vabs_ukernel_fn vabs,
            xnn_init_bf16_default_params_fn init_params = nullptr) const;

  void TestAbs(xnn_f16_vabs_ukernel_fn vabs,
            xnn_init_f16_default_params_fn init_params = nullptr) const;

  void TestAbs(xnn_f32_vabs_ukernel_fn vabs,
            xnn_init_f32_default_params_fn init_params = nullptr) const;

  void Test(xnn_f32_vclamp_ukernel_fn vclamp,
            xnn_init_f32_minmax_params_fn init_params) const;

  void Test(xnn_f16_velu_ukernel_fn velu,
            xnn_init_f16_elu_params_fn init_params) const;

  void Test(xnn_f32_velu_ukernel_fn velu,
            xnn_init_f32_elu_params_fn init_params) const;

  void TestExp(xnn_f32_vexp_ukernel_fn vexp,
            xnn_init_f32_default_params_fn init_params = nullptr) const;

  void TestGelu(xnn_f32_vgelu_ukernel_fn vgelu,
            xnn_init_f32_default_params_fn init_params = nullptr) const;

  void Test(xnn_f16_vhswish_ukernel_fn vhswish,
            xnn_init_f16_hswish_params_fn init_params = nullptr) const;

  void Test(xnn_f32_vhswish_ukernel_fn vhswish,
            xnn_init_f32_hswish_params_fn init_params = nullptr) const;

  void Test(xnn_f16_vlrelu_ukernel_fn vlrelu,
            xnn_init_f16_lrelu_params_fn init_params) const;

  void Test(xnn_f32_vlrelu_ukernel_fn vlrelu,
            xnn_init_f32_lrelu_params_fn init_params) const;

  void TestLog(xnn_f32_vlog_ukernel_fn vlog,
            xnn_init_f32_default_params_fn init_params = nullptr) const;

  void TestNeg(xnn_f16_vneg_ukernel_fn vneg,
            xnn_init_f16_default_params_fn init_params = nullptr) const;

  void TestNeg(xnn_f32_vneg_ukernel_fn vneg,
            xnn_init_f32_default_params_fn init_params = nullptr) const;

  void Test(xnn_f16_vround_ukernel_fn vrnd, OpType op_type,
            xnn_init_f16_rnd_params_fn init_params = nullptr) const;

  void Test(xnn_f32_vround_ukernel_fn vrnd, OpType op_type,
            xnn_init_f32_rnd_params_fn init_params = nullptr) const;

  void Test(xnn_f16_vsigmoid_ukernel_fn vsigmoid,
            xnn_init_f16_sigmoid_params_fn init_params = nullptr) const;

  void Test(xnn_f32_vsigmoid_ukernel_fn vsigmoid,
            xnn_init_f32_sigmoid_params_fn init_params = nullptr) const;

  void TestSqr(xnn_f16_vsqr_ukernel_fn vsqr,
            xnn_init_f16_default_params_fn init_params = nullptr) const;

  void TestSqr(xnn_f32_vsqr_ukernel_fn vsqr,
            xnn_init_f32_default_params_fn init_params = nullptr) const;

  void Test(xnn_f16_vsqrt_ukernel_fn vsqrt,
            xnn_init_f16_sqrt_params_fn init_params = nullptr) const;

  void Test(xnn_f32_vsqrt_ukernel_fn vsqrt,
            xnn_init_f32_sqrt_params_fn init_params = nullptr) const;

  void Test(xnn_f16_vrsqrt_ukernel_fn vrsqrt,
            xnn_init_f16_rsqrt_params_fn init_params = nullptr) const;

  void Test(xnn_f32_vrsqrt_ukernel_fn vrsqrt,
            xnn_init_f32_rsqrt_params_fn init_params = nullptr) const;

  void Test(xnn_f16_vtanh_ukernel_fn vtanh,
            xnn_init_f16_tanh_params_fn init_params = nullptr) const;

  void Test(xnn_f32_vtanh_ukernel_fn vtanh,
            xnn_init_f32_tanh_params_fn init_params = nullptr) const;

  void Test(xnn_f16_vclamp_ukernel_fn vclamp,
            xnn_init_f16_minmax_params_fn init_params) const;

  void Test(xnn_s8_vclamp_ukernel_fn vclamp,
            xnn_init_s8_minmax_params_fn init_params) const;

  void Test(xnn_u8_vclamp_ukernel_fn vclamp,
            xnn_init_u8_minmax_params_fn init_params) const;

  void Test(xnn_u64_u32_vsqrtshift_ukernel_fn vsqrtshift) const;

 private:
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
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(range_min, range_max);

    std::vector<float> x(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(batch_size() +
                         (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<float> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), [&]() { return f32dist(rng); });
      if (inplace()) {
        memcpy(y.data(), x.data(), y.size() * sizeof(float));
      } else {
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
        ASSERT_NEAR(y[i], y_ref[i], tol(y_ref[i]))
            << "at " << i << " / " << batch_size() << ", x[" << i
            << "] = " << std::scientific << x[i];
      }
    }
  }

  union bf16float {
    uint16_t bf[2];
    float f;
  };
  static float cvt_bf16_f32(uint16_t r) {
    bf16float q{};
    q.f = 0;
    q.bf[1] = r;
    return q.f;
  }

  static uint16_t cvt_f32_bf16(float f) {
    bf16float q{};
    q.f = f;
    return q.bf[1];
  }

  // Generic test function for `bf16` `vunary` kernels.
  //
  // The function is templated on the type of the kernel parameters and takes
  // the following arguments:
  //
  //  * `init_params`: A function that populates a given parameters data
  //    structure or returns `nullptr` if there is no default initialization.
  //  * `ref`: A function that computes the reference result for an input `x` of
  //    type `float`, converted from the actual `bf16` input.
  //  * `tol`: A function that computes the absolute tolerance for a reference
  //    result `y_ref` of type `float`. Note that the computed result `y` will
  //    be converted back to `float` for the comparison.
  //  * `range_min`, `range_max`: Limits for the range of input values.
  template <typename UKernelParamsType, typename InitParamsFunc,
            typename ReferenceFunc, typename ToleranceFunc>
  void TestBF16(void (*ukernel)(size_t, const void*, void*,
                                const UKernelParamsType*),
                InitParamsFunc init_params, ReferenceFunc ref,
                ToleranceFunc tol, float range_min, float range_max) const {
    xnnpack::ReplicableRandomDevice rng;
    auto distribution =
        std::uniform_real_distribution<float>(range_min, range_max);
    auto bf16rng = [&]() {
      return cvt_f32_bf16(distribution(rng));
    };

    std::vector<uint16_t> x(batch_size() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> y(
        batch_size() + (inplace() ? XNN_EXTRA_BYTES / sizeof(uint16_t) : 0));
    std::vector<float> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      if (inplace()) {
        std::generate(y.begin(), y.end(), std::ref(bf16rng));
      } else {
        std::generate(x.begin(), x.end(), std::ref(bf16rng));
        std::fill(y.begin(), y.end(), UINT16_C(0xFFFF) /* NaN */);
      }
      const uint16_t* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = ref(cvt_bf16_f32(x_data[i]));
      }

      // Initialize the params.
      UKernelParamsType params;
      const UKernelParamsType* params_ptr = init_params(&params);

      // Call optimized micro-kernel.
      ukernel(batch_size() * sizeof(uint16_t), x_data, y.data(), params_ptr);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_NEAR(cvt_bf16_f32(y[i]), y_ref[i], tol(y_ref[i]))
            << "at " << i << " / " << batch_size() << ", x[" << i
            << "] = " << cvt_bf16_f32(x[i]);
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
    xnnpack::ReplicableRandomDevice rng;
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
        ASSERT_NEAR(fp16_ieee_to_fp32_value(y[i]), y_ref[i], tol(y_ref[i]))
            << "at " << i << " / " << batch_size() << ", x[" << i
            << "] = " << fp16_ieee_to_fp32_value(x[i]);
      }
    }
  }

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
