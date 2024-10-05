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
#include "xnnpack.h"
#include "xnnpack/microfnptr.h"
#include "replicable_random_device.h"

// These help disambiguate Test overloads below.
class Neg {};
class Abs {};
class Log {};
class Sqr {};
class Exp {};
class Gelu {};
class Default {};

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

  void Test(xnn_f32_vrelu_ukernel_fn vrelu,
            xnn_init_f32_relu_params_fn init_params = nullptr,
            Default = Default()) const;

  void Test(xnn_bf16_vabs_ukernel_fn vabs,
            xnn_init_bf16_default_params_fn init_params = nullptr,
            Abs = Abs()) const;

  void Test(xnn_f16_vabs_ukernel_fn vabs,
            xnn_init_f16_default_params_fn init_params = nullptr,
            Abs = Abs()) const;

  void Test(xnn_f32_vabs_ukernel_fn vabs,
            xnn_init_f32_default_params_fn init_params = nullptr,
            Abs = Abs()) const;

  void Test(xnn_f32_vclamp_ukernel_fn vclamp,
            xnn_init_f32_minmax_params_fn init_params,
            Default = Default()) const;

  void Test(xnn_f16_velu_ukernel_fn velu,
            xnn_init_f16_elu_params_fn init_params, Default = Default()) const;

  void Test(xnn_f32_velu_ukernel_fn velu,
            xnn_init_f32_elu_params_fn init_params, Default = Default()) const;

  void Test(xnn_f32_vexp_ukernel_fn vexp,
            xnn_init_f32_default_params_fn init_params = nullptr,
            Exp = Exp()) const;

  void Test(xnn_f32_vgelu_ukernel_fn vgelu,
            xnn_init_f32_default_params_fn init_params = nullptr,
            Gelu = Gelu()) const;

  void Test(xnn_f16_vhswish_ukernel_fn vhswish,
            xnn_init_f16_hswish_params_fn init_params = nullptr,
            Default = Default()) const;

  void Test(xnn_f32_vhswish_ukernel_fn vhswish,
            xnn_init_f32_hswish_params_fn init_params = nullptr,
            Default = Default()) const;

  void Test(xnn_f16_vlrelu_ukernel_fn vlrelu,
            xnn_init_f16_lrelu_params_fn init_params,
            Default = Default()) const;

  void Test(xnn_f32_vlrelu_ukernel_fn vlrelu,
            xnn_init_f32_lrelu_params_fn init_params,
            Default = Default()) const;

  void Test(xnn_f32_vlog_ukernel_fn vlog,
            xnn_init_f32_default_params_fn init_params = nullptr,
            Log = Log()) const;

  void Test(xnn_f16_vneg_ukernel_fn vneg,
            xnn_init_f16_default_params_fn init_params = nullptr,
            Neg = Neg()) const;

  void Test(xnn_f32_vneg_ukernel_fn vneg,
            xnn_init_f32_default_params_fn init_params = nullptr,
            Neg = Neg()) const;

  void Test(xnn_f16_vround_ukernel_fn vrnd, OpType op_type,
            xnn_init_f16_rnd_params_fn init_params = nullptr,
            Default = Default()) const;

  void Test(xnn_f32_vround_ukernel_fn vrnd, OpType op_type,
            xnn_init_f32_rnd_params_fn init_params = nullptr,
            Default = Default()) const;

  void Test(xnn_f16_vsigmoid_ukernel_fn vsigmoid,
            xnn_init_f16_sigmoid_params_fn init_params = nullptr,
            Default = Default()) const;

  void Test(xnn_f32_vsigmoid_ukernel_fn vsigmoid,
            xnn_init_f32_sigmoid_params_fn init_params = nullptr,
            Default = Default()) const;

  void Test(xnn_f16_vsqr_ukernel_fn vsqr,
            xnn_init_f16_default_params_fn init_params = nullptr,
            Sqr = Sqr()) const;

  void Test(xnn_f32_vsqr_ukernel_fn vsqr,
            xnn_init_f32_default_params_fn init_params = nullptr,
            Sqr = Sqr()) const;

  void Test(xnn_f16_vsqrt_ukernel_fn vsqrt,
            xnn_init_f16_sqrt_params_fn init_params = nullptr,
            Default = Default()) const;

  void Test(xnn_f32_vsqrt_ukernel_fn vsqrt,
            xnn_init_f32_sqrt_params_fn init_params = nullptr,
            Default = Default()) const;

  void Test(xnn_f16_vrsqrt_ukernel_fn vrsqrt,
            xnn_init_f16_rsqrt_params_fn init_params = nullptr,
            Default = Default()) const;

  void Test(xnn_f32_vrsqrt_ukernel_fn vrsqrt,
            xnn_init_f32_rsqrt_params_fn init_params = nullptr,
            Default = Default()) const;

  void Test(xnn_f16_vtanh_ukernel_fn vtanh,
            xnn_init_f16_tanh_params_fn init_params = nullptr,
            Default = Default()) const;

  void Test(xnn_f32_vtanh_ukernel_fn vtanh,
            xnn_init_f32_tanh_params_fn init_params = nullptr,
            Default = Default()) const;

  void Test(xnn_f16_vclamp_ukernel_fn vclamp,
            xnn_init_f16_minmax_params_fn init_params,
            Default = Default()) const;

  void Test(xnn_s8_vclamp_ukernel_fn vclamp,
            xnn_init_s8_minmax_params_fn init_params,
            Default = Default()) const;

  void Test(xnn_u8_vclamp_ukernel_fn vclamp,
            xnn_init_u8_minmax_params_fn init_params,
            Default = Default()) const;

 private:
  // Generic test function for `vunary` kernels.
  //
  // The function is templated on the type of the kernel parameters and takes
  // the following arguments:
  //
  //  * `T`: The datatype to test. Should be implicitly convertible to and from
  //    `float`.
  //  * `init_params`: A function that populates a given parameters data
  //    structure or returns `nullptr` if there is no default initialization.
  //  * `ref`: A function that computes the reference result for an input `x`.
  //  * `tol`: A function that computes the absolute tolerance for a reference
  //    result `y_ref`.
  //  * `range_min`, `range_max`: Limits for the range of input values.
  template <typename T, typename UKernelParamsType, typename InitParamsFunc,
            typename ReferenceFunc, typename ToleranceFunc>
  void Test(void (*ukernel)(size_t, const T*, T*,
                                const UKernelParamsType*),
                InitParamsFunc init_params, ReferenceFunc ref,
                ToleranceFunc tol, float range_min, float range_max) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(range_min, range_max);

    std::vector<T> x(batch_size() + XNN_EXTRA_BYTES / sizeof(T));
    std::vector<T> y(batch_size() +
                         (inplace() ? XNN_EXTRA_BYTES / sizeof(T) : 0));
    std::vector<T> y_ref(batch_size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), [&]() { return f32dist(rng); });
      if (inplace()) {
        memcpy(y.data(), x.data(), y.size() * sizeof(T));
      } else {
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const T* x_data = inplace() ? y.data() : x.data();

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        y_ref[i] = ref(x_data[i]);
      }

      // Initialize the params.
      UKernelParamsType params;
      const UKernelParamsType* params_ptr = init_params(&params);

      // Call optimized micro-kernel.
      ukernel(batch_size() * sizeof(T), x_data, y.data(), params_ptr);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        ASSERT_NEAR(y[i], y_ref[i], tol(y_ref[i]))
            << "at " << i << " / " << batch_size() << ", x[" << i
            << "] = " << std::scientific << x[i];
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

// TODO(b/361780131): This could probably be rewritten as some kind of GTest
// instantiate thing instead of macros.
#define XNN_TEST_UNARY_BATCH_EQ(ukernel, arch_flags, batch_tile, datatype, \
                                ...)                                       \
  TEST(ukernel, batch_eq) {                                                \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                  \
    const size_t batch_scale = get_batch_scale<datatype>();                \
    VUnaryMicrokernelTester()                                              \
        .batch_size(batch_tile* batch_scale)                               \
        .Test(__VA_ARGS__);                                                \
  }

#define XNN_TEST_UNARY_BATCH_DIV(ukernel, arch_flags, batch_tile, datatype, \
                                 ...)                                       \
  TEST(ukernel, batch_div) {                                                \
    if (batch_tile == 1) return;                                            \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                   \
    const size_t batch_scale = get_batch_scale<datatype>();                 \
    const size_t batch_step = batch_tile * batch_scale;                     \
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step;  \
         batch_size += batch_step) {                                        \
      VUnaryMicrokernelTester().batch_size(batch_size).Test(__VA_ARGS__);   \
    }                                                                       \
  }

#define XNN_TEST_UNARY_BATCH_LT(ukernel, arch_flags, batch_tile, datatype, \
                                ...)                                       \
  TEST(ukernel, batch_lt) {                                                \
    if (batch_tile == 1) return;                                           \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                  \
    const size_t batch_scale = get_batch_scale<datatype>();                \
    const size_t batch_end = batch_tile * batch_scale;                     \
    for (size_t batch_size = 1; batch_size < batch_end; batch_size++) {    \
      VUnaryMicrokernelTester().batch_size(batch_size).Test(__VA_ARGS__);  \
    }                                                                      \
  }

#define XNN_TEST_UNARY_BATCH_GT(ukernel, arch_flags, batch_tile, datatype, \
                                ...)                                       \
  TEST(ukernel, batch_gt) {                                                \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                  \
    const size_t batch_scale = get_batch_scale<datatype>();                \
    const size_t batch_step = batch_tile * batch_scale;                    \
    const size_t batch_end = batch_tile == 1 ? 10 : 2 * batch_step;        \
    for (size_t batch_size = batch_step + 1; batch_size < batch_end;       \
         batch_size++) {                                                   \
      VUnaryMicrokernelTester().batch_size(batch_size).Test(__VA_ARGS__);  \
    }                                                                      \
  }

#define XNN_TEST_UNARY_INPLACE(ukernel, arch_flags, batch_tile, datatype, ...) \
  TEST(ukernel, inplace) {                                                     \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                      \
    const size_t batch_scale = get_batch_scale<datatype>();                    \
    const size_t batch_end = batch_tile * batch_scale;                         \
    const size_t batch_step = std::max(1, batch_tile - 1);                     \
    for (size_t batch_size = 1; batch_size <= batch_end;                       \
         batch_size += batch_step) {                                           \
      VUnaryMicrokernelTester()                                                \
          .batch_size(batch_size)                                              \
          .inplace(true)                                                       \
          .Test(__VA_ARGS__);                                                  \
    }                                                                          \
  }

#define XNN_TEST_UNARY_QMIN(ukernel, arch_flags, batch_tile, datatype, ...) \
  TEST(ukernel, qmin) {                                                     \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                   \
    const size_t batch_scale = get_batch_scale<datatype>();                 \
    const size_t batch_end = batch_tile * batch_scale;                      \
    const size_t batch_step =                                               \
        batch_scale == 1 ? std::max(1, batch_tile - 1) : batch_end - 1;     \
    for (size_t qmin = 1; qmin < 255; qmin = xnnpack::NextPrime(qmin)) {    \
      for (size_t batch_size = 1; batch_size <= 5 * batch_end;              \
           batch_size += batch_step) {                                      \
        VUnaryMicrokernelTester()                                           \
            .batch_size(batch_size)                                         \
            .qmin(qmin)                                                     \
            .Test(__VA_ARGS__);                                             \
      }                                                                     \
    }                                                                       \
  }

#define XNN_TEST_UNARY_QMAX(ukernel, arch_flags, batch_tile, datatype, ...) \
  TEST(ukernel, qmax) {                                                     \
    TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                   \
    const size_t batch_scale = get_batch_scale<datatype>();                 \
    const size_t batch_end = batch_tile * batch_scale;                      \
    const size_t batch_step =                                               \
        batch_scale == 1 ? std::max(1, batch_tile - 1) : batch_end - 1;     \
    for (size_t qmax = 1; qmax < 255; qmax = xnnpack::NextPrime(qmax)) {    \
      for (size_t batch_size = 1; batch_size <= 5 * batch_end;              \
           batch_size += batch_step) {                                      \
        VUnaryMicrokernelTester()                                           \
            .batch_size(batch_size)                                         \
            .qmax(qmax)                                                     \
            .Test(__VA_ARGS__);                                             \
      }                                                                     \
    }                                                                       \
  }
